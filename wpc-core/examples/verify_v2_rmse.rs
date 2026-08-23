//! RMSE verification for the v2 (6-bit affine) quantization scheme, run
directly against real safetensors weights using the actual production
//! `wpc_core::quant_encoder` code -- not a reimplementation in a scripting
//! language. Reads one named tensor at a time (never mmaps/loads the whole
//! multi-GB file into a Vec) to stay safe on a 15GB-RAM machine.
//!
//! Usage: verify_v2_rmse <safetensors_path> <tensor_name> [<tensor_name> ...]

use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use wpc_core::quant_encoder::affine_quant_block;
use wpc_format::BLOCK_SIZE_V2;

fn read_tensor_f32(mmap: &Mmap, name: &str) -> (Vec<usize>, Vec<f32>) {
    let st = SafeTensors::deserialize(mmap).expect("invalid safetensors file");
    let view = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("tensor {name} not found"));
    let shape = view.shape().to_vec();
    let data = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| {
                let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                f32::from_bits(bits << 16)
            })
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    };
    (shape, data)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: verify_v2_rmse <safetensors_path> <tensor_name> [<tensor_name> ...]");
        std::process::exit(1);
    }
    let path = &args[1];
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    for name in &args[2..] {
        let (shape, data) = read_tensor_f32(&mmap, name);
        let n = data.len();
        let n_blocks = n / BLOCK_SIZE_V2;
        if n_blocks == 0 {
            println!("{name} shape={shape:?}: too small for block_size={BLOCK_SIZE_V2}, skipped");
            continue;
        }
        let usable = n_blocks * BLOCK_SIZE_V2;

        let mut sq_err = 0.0f64;
        let mut sq_sig = 0.0f64;
        for bi in 0..n_blocks {
            let mut block = [0.0f32; BLOCK_SIZE_V2];
            block.copy_from_slice(&data[bi * BLOCK_SIZE_V2..(bi + 1) * BLOCK_SIZE_V2]);
            let qb = affine_quant_block(&block);
            let zp = qb.zero_point.to_f32();
            let sc = qb.scale.to_f32();
            for (&orig_code, &quant_code) in block.iter().zip(qb.codes.iter()) {
                let orig = orig_code as f64;
                let approx = (zp + quant_code as f32 * sc) as f64;
                sq_err += (orig - approx).powi(2);
                sq_sig += orig * orig;
            }
        }
        let rmse = (sq_err / usable as f64).sqrt();
        let sig_rms = (sq_sig / usable as f64).sqrt();
        let rel_pct = 100.0 * rmse / sig_rms;
        println!("{name} shape={shape:?} n_blocks={n_blocks}: {rel_pct:.2}% relative RMSE (via real wpc_core::quant_encoder)");
    }
}
