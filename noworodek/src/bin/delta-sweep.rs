/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 06:40:00
 * REASON FOR CREATION: rank-sweep answered "are trained weights low-rank" with a firm no
 *   -- 0.94 relative error at rank 64 of 2560 on a real Qwen tensor. The question that
 *   result argues for is different: is the CHANGE a fine-tune makes more orderly than the
 *   weights it changes? That is the shape of every claim about editing knowledge by
 *   writing weights, and it is answerable with two models already on this machine.
 * MECHANICS: Reads the same named tensor from two models of identical shape, subtracts,
 *   and runs the same rank sweep on the difference. Crucially it sweeps the tuned weight
 *   as well, at the same ranks, in the same table -- an error figure for a difference
 *   means nothing without the weight's own figure beside it.
 *   Also reports how much of the difference is plausibly format noise rather than
 *   learning: two models stored in different half-precision formats differ slightly
 *   everywhere before any training is considered.
 * SYSTEM PART: Noworodek, weight representation lane.
 * ARCHITECTURE FUNCTION: The follow-on measurement the rank-sweep gate selected. Decides
 *   whether a factored representation is worth building for deltas after it was refused
 *   for raw weights.
 * DEPENDENCIES/LINKS: noworodek::low_rank (low_rank_decompose, rank_metrics),
 *   wpc_runtime::weights::ShardedSafetensors.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-noworodek-2026-08-26\noworodek\src\bin\delta-sweep.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/low-rank-decompose
 * ==========================================
 */

use std::path::PathBuf;
use std::time::Instant;

use noworodek::low_rank::{low_rank_decompose, rank_metrics};
use wpc_runtime::weights::{SafetensorsFile, ShardedSafetensors};

const DEFAULT_RANKS: &[usize] = &[8, 32, 128, 512];

fn usage() -> ! {
    eprintln!(
        "usage: delta-sweep --base <dir> --tuned <dir> --tensor <name> [--ranks 8,32,128]\n\
         \x20      delta-sweep --base <dir> --tuned <dir> --map\n\
         \n\
         --map compares every tensor and reports where the two models differ, without\n\
         factoring anything. Run it first: sweeping a tensor that was never touched\n\
         measures rounding noise and says nothing."
    );
    std::process::exit(2)
}

/// Every tensor name in a model, taken from its shard index.
fn tensor_names(model: &std::path::Path) -> anyhow::Result<Vec<String>> {
    let index = model.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", index.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)?;
    let map = json
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("no weight_map in {}", index.display()))?;
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();
    Ok(names)
}

/// Compare every tensor and report only what actually moved.
///
/// This exists because the first delta sweep was run on a tensor the fine-tune had never
/// touched: the difference was zero and the sweep dutifully factored rounding noise. Find
/// the changed tensors first, then ask questions about them.
fn map_differences(
    base: &std::path::Path,
    tuned: &std::path::Path,
    a: &ShardedSafetensors,
    b: &ShardedSafetensors,
) -> anyhow::Result<()> {
    let names = tensor_names(base)?;
    let tuned_names = tensor_names(tuned)?;
    anyhow::ensure!(
        names == tuned_names,
        "the two models do not hold the same tensors"
    );

    println!("comparing {} tensors", names.len());
    println!();
    println!("  |delta|/|W| | changed % |    |delta| | tensor");
    println!("  ------------+-----------+------------+-------");

    let mut touched = 0usize;
    let mut total_rel = 0.0f64;
    for name in &names {
        let wa = a.read_f32(name);
        let wb = b.read_f32(name);
        if wa.len() != wb.len() {
            println!("  {:>11} | {:>9} | {:>10} | {name}  SHAPE MISMATCH", "-", "-", "-");
            continue;
        }
        let mut sum_sq = 0.0f64;
        let mut changed = 0usize;
        for (&x, &y) in wa.iter().zip(&wb) {
            let d = f64::from(y) - f64::from(x);
            if d != 0.0 {
                changed += 1;
                sum_sq += d * d;
            }
        }
        let norm_delta = sum_sq.sqrt();
        let norm_base = frobenius(&wa);
        let rel = if norm_base > 0.0 { norm_delta / norm_base } else { 0.0 };
        // A tensor moved by less than a thousandth of itself has been re-encoded, not
        // retrained. Printing all 291 rows would bury the handful that matter.
        if rel > 1e-3 {
            touched += 1;
            total_rel += rel;
            println!(
                "  {:>11.6} | {:>8.1}% | {:>10.4} | {name}",
                rel,
                100.0 * changed as f64 / wa.len() as f64,
                norm_delta
            );
        }
    }

    println!();
    if touched == 0 {
        println!("NOTHING moved by more than a thousandth of itself.");
        println!("These two models hold the same weights in different storage formats.");
        println!("There is no fine-tuning difference here to factor.");
    } else {
        println!(
            "{touched} of {} tensors moved meaningfully; mean relative change {:.4}",
            names.len(),
            total_rel / touched as f64
        );
    }
    Ok(())
}

/// Compare two single safetensors files, tensor by tensor.
///
/// A hand-edited model is often one shard rewritten in place rather than a whole model
/// directory with an index, so the sharded path cannot reach it. Only tensors present in
/// both files are compared; anything else is reported rather than skipped silently.
fn map_files(base: &std::path::Path, tuned: &std::path::Path) -> anyhow::Result<()> {
    let a = SafetensorsFile::open(base)?;
    let b = SafetensorsFile::open(tuned)?;
    let mut names = a.names();
    names.sort();
    let in_b: std::collections::HashSet<String> = b.names().into_iter().collect();

    println!("base  : {}  ({} tensors)", base.display(), names.len());
    println!("tuned : {}  ({} tensors)", tuned.display(), in_b.len());
    println!();
    println!("  |delta|/|W| | changed % |    |delta| | shape | tensor");
    println!("  ------------+-----------+------------+-------+-------");

    let mut touched = 0usize;
    let mut total_changed = 0usize;
    let mut missing = 0usize;
    for name in &names {
        if !in_b.contains(name) {
            missing += 1;
            continue;
        }
        let wa = a.read_f32(name);
        let wb = b.read_f32(name);
        if wa.len() != wb.len() {
            println!("  {:>11} | {:>9} | {:>10} |       | {name}  SHAPE MISMATCH", "-", "-", "-");
            continue;
        }
        let mut sum_sq = 0.0f64;
        let mut changed = 0usize;
        for (&x, &y) in wa.iter().zip(&wb) {
            let d = f64::from(y) - f64::from(x);
            if d != 0.0 {
                changed += 1;
                sum_sq += d * d;
            }
        }
        if changed == 0 {
            continue;
        }
        touched += 1;
        total_changed += changed;
        let norm_base = frobenius(&wa);
        let rel = if norm_base > 0.0 { sum_sq.sqrt() / norm_base } else { 0.0 };
        let shape = a.shape(name);
        println!(
            "  {:>11.6} | {:>8.3}% | {:>10.6} | {:?} | {name}",
            rel,
            100.0 * changed as f64 / wa.len() as f64,
            sum_sq.sqrt(),
            shape
        );
    }

    println!();
    println!("{touched} tensors carry a change; {total_changed} values differ in total");
    if missing > 0 {
        println!("{missing} tensors were present in the base file only");
    }
    Ok(())
}

fn frobenius(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

/// Relative error of the best rank-`rank` approximation of `dense`.
fn error_at_rank(dense: &[f32], rows: usize, cols: usize, rank: usize) -> anyhow::Result<f32> {
    let factors = low_rank_decompose(dense, rows, cols, rank)
        .map_err(|e| anyhow::anyhow!("decompose at rank {rank}: {e:?}"))?;
    let metrics = rank_metrics(rows, cols, rank, &factors.materialize(), dense)
        .map_err(|e| anyhow::anyhow!("metrics at rank {rank}: {e:?}"))?;
    Ok(metrics.relative_frobenius_error)
}

fn main() -> anyhow::Result<()> {
    let mut base: Option<PathBuf> = None;
    let mut tuned: Option<PathBuf> = None;
    let mut tensor: Option<String> = None;
    let mut ranks: Vec<usize> = DEFAULT_RANKS.to_vec();
    let mut map = false;
    let mut files = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = args.next().map(PathBuf::from),
            "--tuned" => tuned = args.next().map(PathBuf::from),
            "--tensor" => tensor = args.next(),
            "--map" => map = true,
            "--files" => files = true,
            "--ranks" => {
                let raw = args.next().unwrap_or_default();
                ranks = raw
                    .split(',')
                    .filter_map(|p| p.trim().parse::<usize>().ok())
                    .collect();
            }
            _ => usage(),
        }
    }
    let (Some(base), Some(tuned)) = (base, tuned) else { usage() };

    if files {
        return map_files(&base, &tuned);
    }

    let a = ShardedSafetensors::open(&base)?;
    let b = ShardedSafetensors::open(&tuned)?;

    if map {
        return map_differences(&base, &tuned, &a, &b);
    }
    let Some(tensor) = tensor else { usage() };

    let shape_a = a.shape(&tensor);
    let shape_b = b.shape(&tensor);
    anyhow::ensure!(
        shape_a == shape_b,
        "shapes differ: base {shape_a:?} vs tuned {shape_b:?}; there is no difference to take"
    );
    anyhow::ensure!(shape_a.len() == 2, "{tensor} is not a matrix: {shape_a:?}");
    let (rows, cols) = (shape_a[0], shape_a[1]);

    let t0 = Instant::now();
    let w_base = a.read_f32(&tensor);
    let w_tuned = b.read_f32(&tensor);
    eprintln!("both tensors read in {:?}", t0.elapsed());

    let delta: Vec<f32> = w_base
        .iter()
        .zip(&w_tuned)
        .map(|(&x, &y)| y - x)
        .collect();

    let norm_base = frobenius(&w_base);
    let norm_delta = frobenius(&delta);
    let changed = delta.iter().filter(|&&d| d != 0.0).count();

    // How large a single step in each storage format is, at the magnitudes these weights
    // actually occupy. If the difference is of this order it is a re-encoding, not
    // learning, and no amount of factoring will find structure in it.
    let mean_abs = w_base.iter().map(|&v| f64::from(v.abs())).sum::<f64>() / w_base.len() as f64;
    let f16_step = mean_abs * 2f64.powi(-11); // 10 mantissa bits plus the implicit one
    let bf16_step = mean_abs * 2f64.powi(-8); // 7 mantissa bits plus the implicit one

    println!("tensor            : {tensor}");
    println!("shape             : {rows} x {cols}");
    println!("|W_base|          : {norm_base:.4}");
    println!("|delta|           : {norm_delta:.4}");
    println!(
        "|delta| / |W_base|: {:.4}   <- how much of itself the weight changed",
        norm_delta / norm_base
    );
    println!(
        "elements changed  : {changed} of {} ({:.1}%)",
        rows * cols,
        100.0 * changed as f64 / (rows * cols) as f64
    );
    println!("mean |w|          : {mean_abs:.6}");
    println!(
        "one f16 step      : {f16_step:.8}    one bf16 step: {bf16_step:.8}"
    );
    println!(
        "mean |delta|      : {:.8}",
        delta.iter().map(|&v| f64::from(v.abs())).sum::<f64>() / delta.len() as f64
    );
    println!();
    println!("  rank | error on DELTA | error on TUNED WEIGHT | difference");
    println!("  -----+----------------+-----------------------+-----------");

    for &rank in &ranks {
        if rank == 0 || rank > rows.min(cols) {
            eprintln!("skipping rank {rank}: outside 1..={}", rows.min(cols));
            continue;
        }
        let on_delta = error_at_rank(&delta, rows, cols, rank)?;
        let on_weight = error_at_rank(&w_tuned, rows, cols, rank)?;
        println!(
            "  {:>4} | {:>14.4} | {:>21.4} | {:>+10.4}",
            rank,
            on_delta,
            on_weight,
            on_delta - on_weight
        );
    }

    println!();
    println!("reading it: a difference that is genuinely more orderly than the weight shows");
    println!("a clearly lower error in the left column. Equal columns mean the difference is");
    println!("as unstructured as the weight itself, and factoring it buys nothing.");
    println!();
    println!("if mean |delta| sits near one storage step above, most of what is being");
    println!("measured is a change of number format rather than anything the model learned.");

    Ok(())
}
