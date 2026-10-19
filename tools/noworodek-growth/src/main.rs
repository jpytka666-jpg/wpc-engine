// darkstar-header-v1
// AUTHOR: Klaudia (Claude Code, Consigliere) for Marcin Szul (owner)
// AI MODEL / AGENT: Claude Opus 5.5 (claude-opus-5-5) in Claude Code on ZBook
// TIMESTAMP: 2026-10-18T19:10+01:00 (created)
// REASON: owner "run Noworodek's training on M60A" — the CPU trainer (train-cbms) learns from ONE example per step at ~3 s/step
// MECHANICS: the same 1-layer decoder (RMSNorm, causal single-head attention, SiLU MLP, lm_head, CE on the last position) and the same NWRD
//            weight file as wpc-engine train-cbms, trained on the GPU with mini-batches (mean gradient, SGD). Same data split (last tenth held out),
//            plus a much larger fixed held-out set for honest measurement. Prints the train-cbms keyword lines so the training loop can parse it.
//            --check: one step on one window, prints loss before/after (parity check against the CPU reference).
// SYSTEM PART: AIONS Noworodek / Mikernel Codeone — GPU trainer (Tesla M60, compute 5.2)
// ARCHITECTURE FUNCTION: drop-in replacement for train-cbms in ~/.local/share/aions/noworodek-train/train-loop.mjs
// DEPENDENCIES / LINKS: cudarc 0.19 (dynamic loading), NVRTC 12.8 + cuBLAS 12 (/opt/aions-cuda/lib on Darkstar), src/kernels.cu
// TECH STACK: Rust 2021, CUDA via NVRTC
// LOCAL WORKSPACE: ~/noworodek/gpu-trainer
// GIT / GITHUB: PENDING
// REVISIONS / EVIDENCE: PENDING
use cudarc::cublas::{sys::cublasOperation_t as Op, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, Box<dyn std::error::Error>>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n.max(1) as u64) as usize }
}
fn arg(a: &[String], n: &str) -> Option<String> { a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned() }

/// NWRD v1: "NWRD", u32 version, u32 count, then per tensor: u32 name len, name, u64 n, n×f32 (little endian). Order is kept.
fn load_nwrd(p: &str) -> R<Vec<(String, Vec<f32>)>> {
    let b = std::fs::read(p)?;
    if b.len() < 12 || &b[..4] != b"NWRD" || u32::from_le_bytes(b[4..8].try_into()?) != 1 { return Err("not a Noworodek NWRD v1 file".into()); }
    let declared = u32::from_le_bytes(b[8..12].try_into()?) as usize;
    let (mut at, mut out) = (12usize, Vec::new());
    while at < b.len() {
        let n = u32::from_le_bytes(b[at..at + 4].try_into()?) as usize; at += 4;
        let name = String::from_utf8_lossy(&b[at..at + n]).to_string(); at += n;
        let c = u64::from_le_bytes(b[at..at + 8].try_into()?) as usize; at += 8;
        let v: Vec<f32> = b[at..at + c * 4].chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect(); at += c * 4;
        out.push((name, v));
    }
    if out.len() != declared { return Err(format!("file declares {declared} tensors, read {}", out.len()).into()); }
    Ok(out)
}
fn save_nwrd(p: &str, t: &[(String, Vec<f32>)]) -> R<()> {
    let mut o = Vec::new(); o.extend_from_slice(b"NWRD"); o.extend_from_slice(&1u32.to_le_bytes()); o.extend_from_slice(&(t.len() as u32).to_le_bytes());
    for (n, v) in t { o.extend_from_slice(&(n.len() as u32).to_le_bytes()); o.extend_from_slice(n.as_bytes()); o.extend_from_slice(&(v.len() as u64).to_le_bytes()); for x in v { o.extend_from_slice(&x.to_le_bytes()); } }
    let tmp = format!("{p}.tmp"); std::fs::write(&tmp, o)?; std::fs::rename(tmp, p)?; Ok(())
}

struct Gpu { stream: Arc<CudaStream>, blas: CudaBlas, f: BTreeMap<&'static str, CudaFunction>, _m: Arc<CudaModule> }
impl Gpu {
    fn new() -> R<Self> {
        let ctx = CudaContext::new(0)?; let stream = ctx.default_stream();
        let ptx = compile_ptx_with_opts(include_str!("kernels.cu"), CompileOptions { arch: Some("compute_52"), ..Default::default() })?;
        let m = ctx.load_module(ptx)?; let mut f = BTreeMap::new();
        for n in ["gather_rows", "rms_fwd", "rms_bwd", "attn_fwd", "attn_bwd", "add", "add3", "silu_mul", "silu_mul_bwd", "take_last", "put_last", "ce_rows", "sgd", "emb_sgd", "zero"] { f.insert(n, m.load_function(n)?); }
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Gpu { stream, blas, f, _m: m })
    }
    fn z(&self, n: usize) -> R<CudaSlice<f32>> { Ok(self.stream.alloc_zeros::<f32>(n.max(1))?) }
    /// row-major C[M,N] = op(A) @ op(B) (+ beta*C); A is [M,K] (or [K,M] if ta), B is [K,N] (or [N,K] if tb)
    fn mm(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, m: usize, k: usize, n: usize, ta: bool, tb: bool, beta: f32) -> R<()> {
        let cfg = GemmConfig { transa: if tb { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N }, transb: if ta { Op::CUBLAS_OP_T } else { Op::CUBLAS_OP_N },
            m: n as i32, n: m as i32, k: k as i32, alpha: 1.0f32, lda: if tb { k } else { n } as i32, ldb: if ta { m } else { k } as i32, beta, ldc: n as i32 };
        unsafe { self.blas.gemm(cfg, b, a, c)?; } Ok(())
    }
    fn n1(n: usize) -> LaunchConfig { LaunchConfig { grid_dim: (((n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 } }
    fn rows(r: usize, t: usize) -> LaunchConfig { LaunchConfig { grid_dim: (r as u32, 1, 1), block_dim: (t as u32, 1, 1), shared_mem_bytes: 0 } }
}

const EPS: f32 = 1e-5;
struct Model { v: usize, h: usize, i: usize, names: Vec<String>, w: BTreeMap<String, CudaSlice<f32>> }
fn key(names: &[String], suffix: &str) -> String { names.iter().find(|n| n.ends_with(suffix)).cloned().unwrap_or_else(|| panic!("weight {suffix} missing")) }

/// all activations for one batch of windows, kept for backward
struct Acts { ids: CudaSlice<u32>, tgt: CudaSlice<u32>, x0: CudaSlice<f32>, n1: CudaSlice<f32>, inv1: CudaSlice<f32>, q: CudaSlice<f32>, k: CudaSlice<f32>, v: CudaSlice<f32>,
  a: CudaSlice<f32>, p: CudaSlice<f32>, proj: CudaSlice<f32>, r: CudaSlice<f32>, n2: CudaSlice<f32>, inv2: CudaSlice<f32>, g: CudaSlice<f32>, u: CudaSlice<f32>, sg: CudaSlice<f32>,
  m: CudaSlice<f32>, d: CudaSlice<f32>, y: CudaSlice<f32>, yl: CudaSlice<f32>, fl: CudaSlice<f32>, invf: CudaSlice<f32>, logits: CudaSlice<f32>, loss: CudaSlice<f32> }

fn alloc(g: &Gpu, b: usize, s: usize, md: &Model) -> R<Acts> {
    let (rows, h, i, v) = (b * s, md.h, md.i, md.v);
    Ok(Acts { ids: g.stream.alloc_zeros::<u32>(rows)?, tgt: g.stream.alloc_zeros::<u32>(b)?, x0: g.z(rows * h)?, n1: g.z(rows * h)?, inv1: g.z(rows)?, q: g.z(rows * h)?, k: g.z(rows * h)?, v: g.z(rows * h)?,
        a: g.z(rows * h)?, p: g.z(b * s * s)?, proj: g.z(rows * h)?, r: g.z(rows * h)?, n2: g.z(rows * h)?, inv2: g.z(rows)?, g: g.z(rows * i)?, u: g.z(rows * i)?, sg: g.z(rows * i)?,
        m: g.z(rows * i)?, d: g.z(rows * h)?, y: g.z(rows * h)?, yl: g.z(b * h)?, fl: g.z(b * h)?, invf: g.z(b)?, logits: g.z(b * v)?, loss: g.z(b)? })
}

/// forward for `b` windows (ids already uploaded); leaves per-window loss in a.loss; if grad, logits := dL (mean over batch)
fn forward(g: &Gpu, md: &Model, a: &mut Acts, b: usize, s: usize, grad: bool) -> R<()> {
    let (rows, h, i, v) = (b * s, md.h, md.i, md.v); let w = |n: &str| &md.w[&key(&md.names, n)];
    unsafe {
        g.stream.launch_builder(&g.f["gather_rows"]).arg(w("embeddings.token.weight")).arg(&a.ids).arg(&mut a.x0).arg(&(rows as i32)).arg(&(h as i32)).launch(Gpu::rows(rows, h))?;
        g.stream.launch_builder(&g.f["rms_fwd"]).arg(&a.x0).arg(w("attention_norm.weight")).arg(&mut a.n1).arg(&mut a.inv1).arg(&(rows as i32)).arg(&(h as i32)).arg(&EPS).launch(Gpu::rows(rows, h))?;
    }
    g.mm(&a.n1, w("q_proj.weight"), &mut a.q, rows, h, h, false, false, 0.0)?;
    g.mm(&a.n1, w("k_proj.weight"), &mut a.k, rows, h, h, false, false, 0.0)?;
    g.mm(&a.n1, w("v_proj.weight"), &mut a.v, rows, h, h, false, false, 0.0)?;
    unsafe { g.stream.launch_builder(&g.f["attn_fwd"]).arg(&a.q).arg(&a.k).arg(&a.v).arg(&mut a.a).arg(&mut a.p).arg(&(s as i32)).arg(&(h as i32)).launch(Gpu::rows(b, 128))?; }
    g.mm(&a.a, w("o_proj.weight"), &mut a.proj, rows, h, h, false, false, 0.0)?;
    unsafe {
        g.stream.launch_builder(&g.f["add"]).arg(&a.x0).arg(&a.proj).arg(&mut a.r).arg(&((rows * h) as i32)).launch(Gpu::n1(rows * h))?;
        g.stream.launch_builder(&g.f["rms_fwd"]).arg(&a.r).arg(w("mlp_norm.weight")).arg(&mut a.n2).arg(&mut a.inv2).arg(&(rows as i32)).arg(&(h as i32)).arg(&EPS).launch(Gpu::rows(rows, h))?;
    }
    g.mm(&a.n2, w("gate_proj.weight"), &mut a.g, rows, h, i, false, false, 0.0)?;
    g.mm(&a.n2, w("up_proj.weight"), &mut a.u, rows, h, i, false, false, 0.0)?;
    unsafe { g.stream.launch_builder(&g.f["silu_mul"]).arg(&a.g).arg(&a.u).arg(&mut a.sg).arg(&mut a.m).arg(&((rows * i) as i32)).launch(Gpu::n1(rows * i))?; }
    g.mm(&a.m, w("down_proj.weight"), &mut a.d, rows, i, h, false, false, 0.0)?;
    unsafe {
        g.stream.launch_builder(&g.f["add"]).arg(&a.r).arg(&a.d).arg(&mut a.y).arg(&((rows * h) as i32)).launch(Gpu::n1(rows * h))?;
        g.stream.launch_builder(&g.f["take_last"]).arg(&a.y).arg(&mut a.yl).arg(&(s as i32)).arg(&(h as i32)).launch(Gpu::rows(b, h))?;
        g.stream.launch_builder(&g.f["rms_fwd"]).arg(&a.yl).arg(w("final_norm.weight")).arg(&mut a.fl).arg(&mut a.invf).arg(&(b as i32)).arg(&(h as i32)).arg(&EPS).launch(Gpu::rows(b, h))?;
    }
    g.mm(&a.fl, w("lm_head.weight"), &mut a.logits, b, h, v, false, true, 0.0)?;                    // [B,H] @ lm_head^T
    let gs = 1.0f32 / b as f32; let gr = grad as i32;
    unsafe { g.stream.launch_builder(&g.f["ce_rows"]).arg(&mut a.logits).arg(&a.tgt).arg(&mut a.loss).arg(&(v as i32)).arg(&gs).arg(&gr).launch(Gpu::rows(b, 256))?; }
    Ok(())
}

/// backward from a.logits (= dL) and SGD update of every weight with learning rate lr
fn backward_sgd(g: &Gpu, md: &mut Model, a: &Acts, b: usize, s: usize, lr: f32) -> R<()> {
    let (rows, h, i, v) = (b * s, md.h, md.i, md.v);
    let names = md.names.clone(); let k = |n: &str| key(&names, n);
    let mut grads: Vec<(String, CudaSlice<f32>)> = Vec::new();
    // lm_head: d[V,H] = dL^T[V,B] @ fl[B,H]; d_fl[B,H] = dL[B,V] @ lm_head[V,H]
    let mut dlm = g.z(v * h)?; g.mm(&a.logits, &a.fl, &mut dlm, v, b, h, true, false, 0.0)?;
    let mut dfl = g.z(b * h)?; g.mm(&a.logits, &md.w[&k("lm_head.weight")], &mut dfl, b, v, h, false, false, 0.0)?;
    grads.push((k("lm_head.weight"), dlm));
    let (mut dyl, mut dwf, mut dy) = (g.z(b * h)?, g.z(h)?, g.z(rows * h)?);
    unsafe {
        g.stream.launch_builder(&g.f["rms_bwd"]).arg(&a.yl).arg(&md.w[&k("final_norm.weight")]).arg(&a.invf).arg(&dfl).arg(&mut dyl).arg(&mut dwf).arg(&(b as i32)).arg(&(h as i32)).launch(Gpu::rows(b, h))?;
        g.stream.launch_builder(&g.f["put_last"]).arg(&dyl).arg(&mut dy).arg(&(s as i32)).arg(&(h as i32)).launch(Gpu::rows(b, h))?;
    }
    grads.push((k("final_norm.weight"), dwf));
    // MLP
    let mut dwd = g.z(i * h)?; g.mm(&a.m, &dy, &mut dwd, i, rows, h, true, false, 0.0)?;
    let mut dm = g.z(rows * i)?; g.mm(&dy, &md.w[&k("down_proj.weight")], &mut dm, rows, h, i, false, true, 0.0)?;
    let (mut dg, mut du) = (g.z(rows * i)?, g.z(rows * i)?);
    unsafe { g.stream.launch_builder(&g.f["silu_mul_bwd"]).arg(&dm).arg(&a.g).arg(&a.u).arg(&a.sg).arg(&mut dg).arg(&mut du).arg(&((rows * i) as i32)).launch(Gpu::n1(rows * i))?; }
    let mut dwg = g.z(h * i)?; g.mm(&a.n2, &dg, &mut dwg, h, rows, i, true, false, 0.0)?;
    let mut dwu = g.z(h * i)?; g.mm(&a.n2, &du, &mut dwu, h, rows, i, true, false, 0.0)?;
    let mut dn2 = g.z(rows * h)?; g.mm(&dg, &md.w[&k("gate_proj.weight")], &mut dn2, rows, i, h, false, true, 0.0)?;
    g.mm(&du, &md.w[&k("up_proj.weight")], &mut dn2, rows, i, h, false, true, 1.0)?;
    let (mut dr2, mut dwn2) = (g.z(rows * h)?, g.z(h)?);
    unsafe { g.stream.launch_builder(&g.f["rms_bwd"]).arg(&a.r).arg(&md.w[&k("mlp_norm.weight")]).arg(&a.inv2).arg(&dn2).arg(&mut dr2).arg(&mut dwn2).arg(&(rows as i32)).arg(&(h as i32)).launch(Gpu::rows(rows, h))?; }
    let mut dr = g.z(rows * h)?;
    unsafe { g.stream.launch_builder(&g.f["add"]).arg(&dy).arg(&dr2).arg(&mut dr).arg(&((rows * h) as i32)).launch(Gpu::n1(rows * h))?; }
    grads.push((k("down_proj.weight"), dwd)); grads.push((k("gate_proj.weight"), dwg)); grads.push((k("up_proj.weight"), dwu)); grads.push((k("mlp_norm.weight"), dwn2));
    // attention
    let mut dwo = g.z(h * h)?; g.mm(&a.a, &dr, &mut dwo, h, rows, h, true, false, 0.0)?;
    let mut da = g.z(rows * h)?; g.mm(&dr, &md.w[&k("o_proj.weight")], &mut da, rows, h, h, false, true, 0.0)?;
    let (mut dq, mut dk, mut dv) = (g.z(rows * h)?, g.z(rows * h)?, g.z(rows * h)?);
    unsafe { g.stream.launch_builder(&g.f["attn_bwd"]).arg(&a.q).arg(&a.k).arg(&a.v).arg(&a.p).arg(&da).arg(&mut dq).arg(&mut dk).arg(&mut dv).arg(&(s as i32)).arg(&(h as i32)).launch(Gpu::rows(b, 128))?; }
    let mut dwq = g.z(h * h)?; g.mm(&a.n1, &dq, &mut dwq, h, rows, h, true, false, 0.0)?;
    let mut dwk = g.z(h * h)?; g.mm(&a.n1, &dk, &mut dwk, h, rows, h, true, false, 0.0)?;
    let mut dwv = g.z(h * h)?; g.mm(&a.n1, &dv, &mut dwv, h, rows, h, true, false, 0.0)?;
    let mut dn1 = g.z(rows * h)?; g.mm(&dq, &md.w[&k("q_proj.weight")], &mut dn1, rows, h, h, false, true, 0.0)?;
    g.mm(&dk, &md.w[&k("k_proj.weight")], &mut dn1, rows, h, h, false, true, 1.0)?;
    g.mm(&dv, &md.w[&k("v_proj.weight")], &mut dn1, rows, h, h, false, true, 1.0)?;
    let (mut dx0n, mut dwn1) = (g.z(rows * h)?, g.z(h)?);
    unsafe { g.stream.launch_builder(&g.f["rms_bwd"]).arg(&a.x0).arg(&md.w[&k("attention_norm.weight")]).arg(&a.inv1).arg(&dn1).arg(&mut dx0n).arg(&mut dwn1).arg(&(rows as i32)).arg(&(h as i32)).launch(Gpu::rows(rows, h))?; }
    let mut dx0 = g.z(rows * h)?;
    unsafe { g.stream.launch_builder(&g.f["add"]).arg(&dr).arg(&dx0n).arg(&mut dx0).arg(&((rows * h) as i32)).launch(Gpu::n1(rows * h))?; }
    grads.push((k("o_proj.weight"), dwo)); grads.push((k("q_proj.weight"), dwq)); grads.push((k("k_proj.weight"), dwk)); grads.push((k("v_proj.weight"), dwv)); grads.push((k("attention_norm.weight"), dwn1));
    // apply (all gradients were computed from the old weights, like the reference)
    for (n, gr) in &grads { let w = md.w.get_mut(n).unwrap(); let len = gr.len() as i32; unsafe { g.stream.launch_builder(&g.f["sgd"]).arg(w).arg(gr).arg(&lr).arg(&len).launch(Gpu::n1(len as usize))?; } }
    let emb = md.w.get_mut(&k("embeddings.token.weight")).unwrap();
    unsafe { g.stream.launch_builder(&g.f["emb_sgd"]).arg(emb).arg(&a.ids).arg(&dx0).arg(&lr).arg(&(rows as i32)).arg(&(h as i32)).launch(Gpu::rows(rows, h))?; }
    Ok(())
}

fn upload_windows(g: &Gpu, a: &mut Acts, data: &[usize], starts: &[usize], s: usize) -> R<()> {
    let mut ids = Vec::with_capacity(starts.len() * s); let mut t = Vec::with_capacity(starts.len());
    for &at in starts { ids.extend(data[at..at + s].iter().map(|&x| x as u32)); t.push(data[at + s] as u32); }
    g.stream.memcpy_htod(&ids, &mut a.ids.slice_mut(0..ids.len()))?; g.stream.memcpy_htod(&t, &mut a.tgt.slice_mut(0..t.len()))?; Ok(())
}
fn eval(g: &Gpu, md: &Model, a: &mut Acts, data: &[usize], starts: &[usize], b: usize, s: usize) -> R<f32> {
    let mut total = 0.0f64;
    for ch in starts.chunks(b) { upload_windows(g, a, data, ch, s)?; forward(g, md, a, ch.len(), s, false)?; let l = g.stream.clone_dtoh(&a.loss)?; total += l[..ch.len()].iter().map(|x| *x as f64).sum::<f64>(); }
    Ok((total / starts.len() as f64) as f32)
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let ids_path = arg(&args, "--ids").ok_or("--ids file.u16 required")?; let load = arg(&args, "--load").ok_or("--load weights.bin required")?;
    let save = arg(&args, "--save"); let steps: usize = arg(&args, "--steps").and_then(|x| x.parse().ok()).unwrap_or(2000);
    let b: usize = arg(&args, "--batch").and_then(|x| x.parse().ok()).unwrap_or(256); let lr: f32 = arg(&args, "--lr").and_then(|x| x.parse().ok()).unwrap_or(0.05);
    let s: usize = arg(&args, "--seq").and_then(|x| x.parse().ok()).unwrap_or(8); let seed: u64 = arg(&args, "--seed").and_then(|x| x.parse().ok()).unwrap_or(7);
    let every: usize = arg(&args, "--eval-every").and_then(|x| x.parse().ok()).unwrap_or(100); let patience: usize = arg(&args, "--patience").and_then(|x| x.parse().ok()).unwrap_or(5);
    let n_eval: usize = arg(&args, "--eval-windows").and_then(|x| x.parse().ok()).unwrap_or(4096);
    let ids: Vec<usize> = std::fs::read(&ids_path)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as usize).collect();
    let t0 = Instant::now();
    let tensors = load_nwrd(&load)?; let names: Vec<String> = tensors.iter().map(|(n, _)| n.clone()).collect();
    let emb_len = tensors.iter().find(|(n, _)| n.ends_with("embeddings.token.weight")).ok_or("no embedding")?.1.len();
    let h = tensors.iter().find(|(n, _)| n.ends_with("final_norm.weight")).ok_or("no final norm")?.1.len();
    let i = tensors.iter().find(|(n, _)| n.ends_with("gate_proj.weight")).ok_or("no gate")?.1.len() / h; let v = emb_len / h;
    if ids.iter().any(|&x| x >= v) { return Err(format!("an id in {ids_path} is >= vocab {v}").into()); }
    let g = Gpu::new()?;
    let mut w = BTreeMap::new(); for (n, vals) in &tensors { w.insert(n.clone(), g.stream.clone_htod(vals)?); }
    let mut md = Model { v, h, i, names: names.clone(), w };
    let split = ids.len() * 9 / 10; let (train, held) = ids.split_at(split);
    // held-out windows: the 24 of train-cbms (seed ^ 0xA5A5) for compatibility, and a large fixed set for an honest number
    let mut hr = Rng(seed ^ 0xA5A5); let held24: Vec<usize> = (0..24).map(|_| hr.below(held.len() - s - 1)).collect();
    let mut hr2 = Rng(seed ^ 0x5A5A5A); let n_e = n_eval.min(held.len() - s - 1); let held_big: Vec<usize> = (0..n_e).map(|_| hr2.below(held.len() - s - 1)).collect();
    let mut counts = vec![0usize; v]; for &x in train { counts[x] += 1; }
    let unigram = held_big.iter().map(|&at| { let p = (counts[held[at + s]] as f64 + 1.0) / (train.len() as f64 + v as f64); -p.ln() }).sum::<f64>() / n_e as f64;
    let mut a = alloc(&g, b, s, &md)?;
    println!("NOWORODEK CBMS TRAINING V1 — GPU (Tesla M60 via CUDA/NVRTC, batch {b})");
    println!("korpus         : {} numerow ({} uczace, {} odlozone)", ids.len(), train.len(), held.len());
    println!("slownik        : {v}"); println!("model          : hidden {h}, inter {i}, warstw 1, sekwencja {s}");
    println!("krokow         : {steps}, batch {b}, lr {lr}, ziarno {seed}, ocena na {n_e} oknach odlozonych");
    println!("PROG 1, kostka : {:.4}", (v as f32).ln()); println!("PROG 2, czestosc: {unigram:.4}");
    let e24 = eval(&g, &md, &mut a, held, &held24, b, s)?; let e0 = eval(&g, &md, &mut a, held, &held_big, b, s)?;
    println!("wczytano       : {} tensorow z {load}", tensors.len());
    println!("zgodnosc24 po wczytaniu: {e24:.4}   (24 okna jak train-cbms)");
    println!("odlozone po wczytaniu: {e0:.4}");
    if args.iter().any(|x| x == "--check") {                               // parity: one step on the first training window, batch 1
        let at = arg(&args, "--check-at").and_then(|x| x.parse().ok()).unwrap_or(0usize);
        let mut a1 = alloc(&g, 1, s, &md)?; upload_windows(&g, &mut a1, train, &[at], s)?;
        forward(&g, &md, &mut a1, 1, s, true)?; let before = g.stream.clone_dtoh(&a1.loss)?[0];
        backward_sgd(&g, &mut md, &a1, 1, s, lr)?; forward(&g, &md, &mut a1, 1, s, false)?; let after = g.stream.clone_dtoh(&a1.loss)?[0];
        println!("CHECK window {at}: loss before {before:.6}  after one step (lr {lr}) {after:.6}"); return Ok(());
    }
    let mut rng = Rng(seed | 1); let (mut best, mut since, mut best_w) = (e0, 0usize, None::<Vec<(String, Vec<f32>)>>);
    let (mut win, mut win_n) = (0.0f64, 0usize);
    for step in 1..=steps {
        let starts: Vec<usize> = (0..b).map(|_| rng.below(train.len() - s - 1)).collect();
        upload_windows(&g, &mut a, train, &starts, s)?; forward(&g, &md, &mut a, b, s, true)?;
        if step % 10 == 0 { let l = g.stream.clone_dtoh(&a.loss)?; win += l.iter().map(|x| *x as f64).sum::<f64>() / b as f64; win_n += 1; }
        backward_sgd(&g, &mut md, &a, b, s, lr)?;
        if step % every == 0 || step == steps {
            let e = eval(&g, &md, &mut a, held, &held_big, b, s)?;
            println!("krok {step:6} | uczace {:7.4} | odlozone {e:7.4} | {:6.1} ms/krok", if win_n > 0 { win / win_n as f64 } else { f64::NAN }, t0.elapsed().as_millis() as f64 / step as f64);
            win = 0.0; win_n = 0;
            if e < best - 1e-4 { best = e; since = 0; best_w = Some(names.iter().map(|n| Ok((n.clone(), g.stream.clone_dtoh(&md.w[n])?))).collect::<R<Vec<_>>>()?); }
            else { since += 1; if patience > 0 && since >= patience { println!("STOP: odlozone nie poprawilo sie przez {patience} pomiarow (najlepsze {best:.4})"); break; } }
        }
    }
    if let Some(bw) = &best_w { for (n, vals) in bw { md.w.insert(n.clone(), g.stream.clone_htod(vals)?); } }
    let fin = eval(&g, &md, &mut a, held, &held_big, b, s)?; let fin24 = eval(&g, &md, &mut a, held, &held24, b, s)?;
    if let Some(p) = save { let out: Vec<(String, Vec<f32>)> = names.iter().map(|n| Ok((n.clone(), g.stream.clone_dtoh(&md.w[n])?))).collect::<R<Vec<_>>>()?; save_nwrd(&p, &out)?; println!("zapisano       : {} tensorow do {p}", out.len()); }
    println!("zgodnosc24 PO : {fin24:.4}"); println!("odlozone PO    : {fin:.4}");
    println!("czas calkowity : {:.1} s", t0.elapsed().as_secs_f32());
    println!("{}", if (fin as f64) < unigram { "WYNIK: BIJE PROG CZESTOSCI — model uzywa kontekstu." } else { "WYNIK: jeszcze nie bije progu czestosci." });
    Ok(())
}
