// AIONS / Noworodek growing brain — isolated candidate, never writes live best.bin.
// NWRD v1 MLP widening. Preserve ALL old weights; append new neurons with nonzero
// gate/up and zero down output, so forward output is initially unchanged.
// AUTHOR: Nez / GPT-6. Owner approval 2026-10-19. Rust-only, ZBOOK lab.
use std::{
    collections::HashSet,
    env,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, Debug)]
struct Tensor {
    name: String,
    data: Vec<f32>,
}
fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}
fn load(r: &mut impl Read) -> Result<Vec<Tensor>, String> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"NWRD" || read_u32(r)? != 1 {
        return Err("unsupported NWRD magic/version".into());
    }
    let n = read_u32(r)? as usize;
    if n == 0 || n > 1000 {
        return Err("invalid tensor count".into());
    }
    let mut a = Vec::with_capacity(n);
    let mut names = HashSet::new();
    for _ in 0..n {
        let name_len = read_u32(r)? as usize;
        if name_len == 0 || name_len > 1024 {
            return Err("invalid name length".into());
        }
        let mut s = vec![0u8; name_len];
        r.read_exact(&mut s).map_err(|e| e.to_string())?;
        let name = String::from_utf8(s).map_err(|e| e.to_string())?;
        if !names.insert(name.clone()) {
            return Err(format!("duplicate tensor {name}"));
        }
        let count = read_u64(r)?;
        if count > 80_000_000 {
            return Err(format!("tensor too large: {name}"));
        }
        let len = usize::try_from(count).map_err(|e| e.to_string())?;
        let mut raw = vec![0u8; len.checked_mul(4).ok_or("size overflow")?];
        r.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if data.iter().any(|x| !x.is_finite()) {
            return Err(format!("NaN or infinite weight in {name}"));
        }
        a.push(Tensor { name, data });
    }
    let mut extra = [0u8; 1];
    if r.read(&mut extra).map_err(|e| e.to_string())? != 0 {
        return Err("trailing bytes".into());
    }
    Ok(a)
}
fn write(t: &[Tensor], w: &mut impl Write) -> io::Result<()> {
    w.write_all(b"NWRD")?;
    w.write_all(&1u32.to_le_bytes())?;
    w.write_all(&(t.len() as u32).to_le_bytes())?;
    for x in t {
        w.write_all(&(x.name.len() as u32).to_le_bytes())?;
        w.write_all(x.name.as_bytes())?;
        w.write_all(&(x.data.len() as u64).to_le_bytes())?;
        for v in &x.data {
            w.write_all(&v.to_le_bytes())?
        }
    }
    Ok(())
}
fn idx(t: &[Tensor], ending: &str) -> Result<usize, String> {
    let matches: Vec<usize> = t
        .iter()
        .enumerate()
        .filter(|(_, x)| x.name.ends_with(ending))
        .map(|(i, _)| i)
        .collect();
    if matches.len() != 1 {
        return Err(format!(
            "need exactly one tensor ending {ending}; found {}",
            matches.len()
        ));
    }
    Ok(matches[0])
}
fn param_count(t: &[Tensor]) -> usize {
    t.iter().map(|x| x.data.len()).sum()
}
#[derive(Debug)]
struct Growth {
    old_inter: usize,
    new_inter: usize,
    hidden: usize,
    old_count: usize,
    new_count: usize,
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}
fn grow(t: &mut [Tensor], new_inter: usize, seed: u64) -> Result<Growth, String> {
    let q = idx(t, "layers.00.mlp.gate_proj.weight")?;
    let u = idx(t, "layers.00.mlp.up_proj.weight")?;
    let d = idx(t, "layers.00.mlp.down_proj.weight")?;
    let hn = idx(t, "final_norm.weight")?;
    // This first-stage grower supports one transformer layer only.
    if t.iter().any(|x| x.name.contains("layers.01.")) {
        return Err("multi-layer growth needs explicit mapping".into());
    }
    let h = t[hn].data.len();
    if h == 0 || t[q].data.len() % h != 0 {
        return Err("invalid hidden/intermediate dimensions".into());
    }
    let old = t[q].data.len() / h;
    if t[u].data.len() != old * h || t[d].data.len() != h * old {
        return Err("MLP tensor size mismatch".into());
    }
    // No permanent brain-size ceiling: each validated step can grow at most 2x.
    // Resource/quality gates belong to the promotion controller, not the tensor format.
    if new_inter <= old || new_inter > old.saturating_mul(2).max(128) || new_inter % 128 != 0 {
        return Err(format!(
            "target intermediate must be > {old}, <= 2x current size per step, and divisible by 128"
        ));
    }
    // GPU runtime sees gate/up as row-major [hidden, intermediate].
    // Preserve each old ROW (not one contiguous prefix), then add new neurons.
    let before = param_count(t);
    for (k, ix) in [q, u].into_iter().enumerate() {
        let mut r = Rng(seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(k as u64 + 1)));
        if r.0 == 0 {
            r.0 = 1;
        }
        let scale = (1.0f32 / (h as f32).sqrt()) * 0.5;
        let original = std::mem::take(&mut t[ix].data);
        let mut widened = vec![0.0f32; h * new_inter];
        for row in 0..h {
            widened[row * new_inter..row * new_inter + old]
                .copy_from_slice(&original[row * old..row * old + old]);
            for col in old..new_inter {
                widened[row * new_inter + col] = r.next() * scale;
            }
        }
        t[ix].data = widened;
    }
    // GPU runtime sees down as row-major [intermediate, hidden].
    // Adding zero-output neurons leaves EVERY original output unchanged.
    t[d].data.resize(new_inter * h, 0.0);
    Ok(Growth {
        old_inter: old,
        new_inter,
        hidden: h,
        old_count: before,
        new_count: param_count(t),
    })
}
fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7
        || args[1] != "--source"
        || args[3] != "--output"
        || args[5] != "--target-inter"
    {
        return Err(
            "usage: noworodek-grow --source EXISTING.bin --output LAB-ONLY.bin --target-inter 512"
                .into(),
        );
    }
    let (src, dst) = (Path::new(&args[2]), Path::new(&args[4]));
    let target: usize = args[6].parse().map_err(|_| "invalid target-inter")?;
    if dst.exists()
        || dst.file_name().is_some_and(|n| n == "best.bin")
        || dst.to_string_lossy().contains("/noworodek-train/")
    {
        return Err("REFUSED: output must be new, outside live trainer, and not best.bin".into());
    }
    if !src.is_file() {
        return Err("source not a file".into());
    }
    let mut t = load(&mut BufReader::new(
        File::open(src).map_err(|e| e.to_string())?,
    ))?;
    let stats = grow(&mut t, target, 0x41_49_4f_4e_53_2d_4e_57)?;
    let tmp = dst.with_extension(format!("tmp-{}", std::process::id()));
    let result = (|| -> Result<(), String> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| e.to_string())?;
        let mut out = BufWriter::new(file);
        write(&t, &mut out).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        let file = out.into_inner().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        // Read it back and validate before publishing.
        let verified = load(&mut BufReader::new(
            File::open(&tmp).map_err(|e| e.to_string())?,
        ))?;
        if param_count(&verified) != stats.new_count {
            return Err("self-validation failed".into());
        }
        fs::rename(&tmp, dst).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result?;
    println!(
        "PASS: growth candidate {:?}; original {:?}; hidden {}; intermediate {} -> {}; parameters {} -> {} (+{})",
        dst,
        src,
        stats.hidden,
        stats.old_inter,
        stats.new_inter,
        stats.old_count,
        stats.new_count,
        stats.new_count - stats.old_count
    );
    println!(
        "LIVE best.bin untouched. Candidate requires independent GPU evaluation before promotion."
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {e}");
        std::process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Vec<Tensor> {
        let h = 4;
        let old = 8;
        vec![
            Tensor {
                name: "model.final_norm.weight".into(),
                data: vec![1.0; h],
            },
            Tensor {
                name: "model.layers.00.mlp.gate_proj.weight".into(),
                data: (0..old * h).map(|x| x as f32 * 0.01).collect(),
            },
            Tensor {
                name: "model.layers.00.mlp.up_proj.weight".into(),
                data: (0..old * h).map(|x| x as f32 * 0.02).collect(),
            },
            Tensor {
                name: "model.layers.00.mlp.down_proj.weight".into(),
                data: (0..old * h).map(|x| x as f32 * 0.03).collect(),
            },
        ]
    }
    fn eval(t: &[Tensor], x: &[f32]) -> Vec<f32> {
        let h = x.len();
        let i = t[idx(t, "mlp.up_proj.weight").unwrap()].data.len() / h;
        let gate = &t[idx(t, "mlp.gate_proj.weight").unwrap()].data;
        let up = &t[idx(t, "mlp.up_proj.weight").unwrap()].data;
        let down = &t[idx(t, "mlp.down_proj.weight").unwrap()].data;
        let mut a = vec![0f32; i];
        for k in 0..i {
            let g = (0..h).map(|j| x[j] * gate[j * i + k]).sum::<f32>();
            let u = (0..h).map(|j| x[j] * up[j * i + k]).sum::<f32>();
            a[k] = (g / (1.0 + (-g).exp())) * u;
        }
        (0..h)
            .map(|row| (0..i).map(|k| a[k] * down[k * h + row]).sum())
            .collect()
    }
    #[test]
    fn growth_preserves_old_mlp_output() {
        let old = sample();
        let x = [0.5, -1.0, 0.25, 2.0];
        let baseline = eval(&old, &x);
        let mut expanded = old.clone();
        let s = grow(&mut expanded, 128, 47).unwrap();
        assert_eq!((s.old_count, s.new_count), (100, 1540));
        let after = eval(&expanded, &x);
        for (a, b) in baseline.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-6)
        }
        let new_i = 128;
        let down = &expanded[idx(&expanded, "mlp.down_proj.weight").unwrap()].data;
        assert!(down[8 * 4..].iter().all(|x| *x == 0.0));
        let up = &expanded[idx(&expanded, "mlp.up_proj.weight").unwrap()].data;
        for row in 0..4 {
            assert!(
                up[row * new_i + 8..(row + 1) * new_i]
                    .iter()
                    .any(|x| *x != 0.0)
            );
        }
        let mut altered = expanded.clone();
        let di = idx(&altered, "mlp.down_proj.weight").unwrap();
        for row in 0..4 {
            altered[di].data[8 * 4 + row] = 1.0
        }
        assert_ne!(
            eval(&altered, &x),
            after,
            "new paths must influence outputs when trained"
        );
    }
    #[test]
    fn serialization_roundtrip_and_determinism() {
        let mut a = sample();
        let mut b = a.clone();
        grow(&mut a, 128, 7).unwrap();
        grow(&mut b, 128, 7).unwrap();
        assert_eq!(
            a.iter().map(|x| &x.data).collect::<Vec<_>>(),
            b.iter().map(|x| &x.data).collect::<Vec<_>>()
        );
        let mut bytes = Vec::new();
        write(&a, &mut bytes).unwrap();
        let read = load(&mut io::Cursor::new(bytes.clone())).unwrap();
        assert_eq!(param_count(&read), param_count(&a));
        bytes.push(1);
        assert!(load(&mut io::Cursor::new(bytes)).is_err());
    }
    #[test]
    fn repeated_growth_preserves_old_mlp_output() {
        let mut t = sample();
        let x = [-0.3, 0.9, -0.7, 1.1];
        let original = eval(&t, &x);
        grow(&mut t, 128, 10).unwrap();
        let second = grow(&mut t, 256, 11).unwrap();
        assert_eq!((second.old_inter, second.new_inter), (128, 256));
        for (a, b) in original.iter().zip(eval(&t, &x)) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn growth_rejects_invalid_target() {
        let mut t = sample();
        assert!(grow(&mut t, 8, 1).is_err());
        assert!(grow(&mut t, 129, 1).is_err());
    }
}
