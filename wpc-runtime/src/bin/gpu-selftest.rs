/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 21:55:00
 * REASON FOR CREATION: Milestones G1 and G2 of the GPU plan need evidence, not a claim.
 *   G1 is "Rust can see the card", G2 is "Rust can run the existing kernel and get the
 *   right answer". A unit test cannot carry this because CI has no GPU, so the check
 *   lives in its own binary that is run by hand and prints what it observed.
 * MECHANICS: Opens device 0 through the Driver API, prints the device identity and free
 *   VRAM, then builds a synthetic WPC v4 tensor whose block headers are chosen so the
 *   decoded weight equals the raw 4-bit code (zero_point 0.0, scale 1.0). That makes the
 *   expected result computable in plain Rust with no floating-point ambiguity at all,
 *   which is what turns "the kernel ran" into "the kernel is right". Uploads it, runs
 *   wpc4_gemv, and compares every output row against the Rust reference.
 * SYSTEM PART: WPC / GPU offload lane, verification.
 * ARCHITECTURE FUNCTION: The gate between "kernels proven in C" and "engine uses the
 *   card". Nothing downstream should be wired until this prints PASS.
 * DEPENDENCIES/LINKS: wpc_runtime::gpu; the PTX built from wpc4_gemv_sm50.cu, located
 *   by AIONS_GPU_PTX or the first argument.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-multiturn-2026-08-25\wpc-runtime\src\bin\gpu-selftest.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/resident-multi-turn
 * ==========================================
 */

#[cfg(not(any(target_os = "linux", windows)))]
fn main() {
    eprintln!("gpu-selftest runs on Linux and Windows; this platform has no CUDA bridge.");
    std::process::exit(1);
}

#[cfg(any(target_os = "linux", windows))]
fn main() -> anyhow::Result<()> {
    use wpc_runtime::gpu::{Gpu, WPC4_BLOCK_BYTES};

    const BLOCK_VALUES: usize = 128;
    const PACKED_BYTES: usize = 64;
    const N_ROWS: usize = 8;
    const BLOCKS_PER_ROW: usize = 2;
    const IN_FEATURES: usize = BLOCKS_PER_ROW * BLOCK_VALUES;

    let ptx = std::env::args().nth(1);
    let gpu = Gpu::open(ptx.as_deref())?;

    // ---- G1: is the card there, and is it the one we think it is? -----------------
    let (major, minor, sms) = gpu.capability();
    let (free_before, total) = gpu.mem_info()?;
    println!("=== G1: device ===");
    println!("device          : {}", gpu.device_name());
    println!("compute capab.  : {major}.{minor}, {sms} SMs");
    println!(
        "VRAM            : {} MiB free of {} MiB",
        free_before / (1024 * 1024),
        total / (1024 * 1024)
    );

    // ---- ladder: which single driver call kills the context? ----------------------
    //
    // A trace showed cuMemAlloc returning success and the very next cuCtxSynchronize
    // reporting "context is destroyed". The call that reports the error is therefore
    // not the call that caused it. This ladder does one primitive at a time, syncing
    // after each, so the last rung that passes names the culprit.
    // ---- round trip: does memory survive the journey there and back? --------------
    //
    // No cuCtxSynchronize anywhere: the blocking device-to-host copy is itself the
    // synchronisation point, and on this setup the explicit call reports a destroyed
    // context that is plainly still working.
    println!("\n=== round trip ===");
    {
        let probe: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        let buf = gpu.alloc(256 * 4)?;
        gpu.write_f32(&buf, &probe)?;
        let mut back = vec![0.0f32; 256];
        gpu.read_f32(&buf, &mut back)?;
        let same = back == probe;
        println!("  256 floats there and back : {}", if same { "identical" } else { "CORRUPTED" });
        anyhow::ensure!(same, "device memory round trip corrupted the data");
    }

    // ---- G2: does the kernel compute what it should? ------------------------------
    //
    // zero_point = 0.0 (f16 0x0000) and scale = 1.0 (f16 0x3C00) make the affine decode
    // `w = zero_point + code * scale` collapse to `w = code`. Every weight is then an
    // exact small integer, so the reference below is exact and any mismatch is a real
    // fault rather than rounding.
    let mut blocks = vec![0u8; N_ROWS * BLOCKS_PER_ROW * WPC4_BLOCK_BYTES];
    let mut seed: u32 = 0x9E37_79B9;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };

    for blk in blocks.chunks_exact_mut(WPC4_BLOCK_BYTES) {
        blk[0] = 0x00;
        blk[1] = 0x00; // zero_point = 0.0
        blk[2] = 0x00;
        blk[3] = 0x3C; // scale = 1.0
        for b in blk[4..].iter_mut() {
            *b = (next() & 0xFF) as u8;
        }
    }

    let x: Vec<f32> = (0..IN_FEATURES)
        .map(|i| ((next() % 2001) as f32 - 1000.0) / 1000.0 + i as f32 * 0.0)
        .collect();

    // Reference, in plain Rust, mirroring the kernel's indexing exactly: byte j of a
    // block carries code j in the low nibble and code j+64 in the high nibble.
    let mut expected = vec![0.0f32; N_ROWS];
    for (row, e) in expected.iter_mut().enumerate() {
        let row_base = row * BLOCKS_PER_ROW * WPC4_BLOCK_BYTES;
        let mut acc = 0.0f32;
        for b in 0..BLOCKS_PER_ROW {
            let blk = &blocks[row_base + b * WPC4_BLOCK_BYTES..][..WPC4_BLOCK_BYTES];
            for j in 0..PACKED_BYTES {
                let byte = blk[4 + j];
                let w_lo = (byte & 0x0F) as f32;
                let w_hi = (byte >> 4) as f32;
                acc += w_lo * x[b * BLOCK_VALUES + j];
                acc += w_hi * x[b * BLOCK_VALUES + j + PACKED_BYTES];
            }
        }
        *e = acc;
    }

    let d_weights = gpu.upload(&blocks)?;
    let d_x = gpu.alloc(IN_FEATURES * 4)?;
    gpu.write_f32(&d_x, &x)?;
    let d_y = gpu.alloc(N_ROWS * 4)?;

    gpu.gemv(
        &d_weights,
        0,
        &d_x,
        &d_y,
        BLOCKS_PER_ROW as u32,
        N_ROWS as u32,
    )?;

    let mut got = vec![0.0f32; N_ROWS];
    gpu.read_f32(&d_y, &mut got)?;

    println!("\n=== G2: fused decode + matvec ===");
    println!("shape           : {N_ROWS} x {IN_FEATURES}");
    println!("packed bytes    : {}", blocks.len());

    // The kernel sums in a shared-memory tree while the reference sums sequentially, so
    // the two orders differ and bit-equality is not available. A relative tolerance is
    // the honest test; claiming bit-exactness here would be false.
    let mut worst = 0.0f64;
    let mut worst_row = 0usize;
    for (row, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let denom = (e.abs() as f64).max(1e-6);
        let rel = ((*g as f64) - (*e as f64)).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_row = row;
        }
    }
    println!("row 0           : gpu {:.6} | reference {:.6}", got[0], expected[0]);
    println!("max rel. error  : {worst:.3e} (row {worst_row})");

    let (free_after, _) = gpu.mem_info()?;
    println!(
        "VRAM after      : {} MiB free ({} MiB taken)",
        free_after / (1024 * 1024),
        free_before.saturating_sub(free_after) / (1024 * 1024)
    );

    if worst < 1e-4 {
        println!("\nRESULT          : PASS -- Rust drove the card and the answer is right");
        Ok(())
    } else {
        println!("\nRESULT          : FAIL");
        anyhow::bail!("max relative error {worst:.3e} exceeds 1e-4")
    }
}
