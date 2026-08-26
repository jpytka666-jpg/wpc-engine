/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 05:10:00
 * REASON FOR CREATION: `low_rank.rs` could store, execute and measure a factored matrix
 *   but nothing produced one, so the representation had never met a real tensor. This
 *   answers the question the module exists to answer: is low-rank a serious second
 *   representation beside WPC, or does it only look good on paper.
 * MECHANICS: Reads one named tensor from a real model, factors it at a sweep of ranks,
 *   reconstructs each, and reports storage, arithmetic and error against the dense
 *   original. The WPC figures for the same tensor are printed alongside as already
 *   measured constants, not re-run, so the comparison has a fixed reference point.
 * SYSTEM PART: Noworodek, weight representation lane.
 * ARCHITECTURE FUNCTION: The evaluation gate demanded by docs/noworodek/design-goals.md:
 *   "No representation is accepted as a performance improvement without measurements."
 *   Its output decides whether low-rank proceeds or the effort moves to model deltas.
 * DEPENDENCIES/LINKS: noworodek::low_rank (decompose, rank_metrics, LowRankMatrix),
 *   wpc_runtime::weights::ShardedSafetensors for reading a model split across files.
 * TECH STACK: Rust 2021, no new crates beyond rayon already added for the decomposer.
 * LOCAL WORKSPACE: C:\temp\aions-noworodek-2026-08-26\noworodek\src\bin\rank-sweep.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/low-rank-decompose
 * ==========================================
 */

use std::path::PathBuf;
use std::time::Instant;

use noworodek::low_rank::{low_rank_decompose, rank_metrics};
use wpc_runtime::weights::ShardedSafetensors;

const DEFAULT_RANKS: &[usize] = &[8, 16, 32, 64, 128, 256];

fn usage() -> ! {
    eprintln!("usage: rank-sweep --model <dir> --tensor <name> [--ranks 8,16,32]");
    std::process::exit(2)
}

fn main() -> anyhow::Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut tensor: Option<String> = None;
    let mut ranks: Vec<usize> = DEFAULT_RANKS.to_vec();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            "--tensor" => tensor = args.next(),
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
    let Some(model) = model else { usage() };

    let t0 = Instant::now();
    let shards = ShardedSafetensors::open(&model)?;
    eprintln!("model opened in {:?}", t0.elapsed());

    let Some(tensor) = tensor else { usage() };
    let shape = shards.shape(&tensor);
    if shape.len() != 2 {
        anyhow::bail!("{tensor} has shape {shape:?}; a sweep needs a matrix, not a {}-d tensor", shape.len());
    }
    let (rows, cols) = (shape[0], shape[1]);

    let t1 = Instant::now();
    let dense = shards.read_f32(&tensor);
    anyhow::ensure!(
        dense.len() == rows * cols,
        "read {} values for a {rows}x{cols} tensor",
        dense.len()
    );
    eprintln!("tensor read in {:?}", t1.elapsed());

    // How much of the matrix there is to lose, so a relative error has a scale.
    let frobenius: f64 = dense.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();

    println!("tensor          : {tensor}");
    println!("shape           : {rows} x {cols}  ({} values)", rows * cols);
    println!("dense f32       : {:.1} MiB", (rows * cols * 4) as f64 / 1048576.0);
    println!("Frobenius norm  : {frobenius:.4}");
    println!();
    println!("  rank | factors MiB | vs dense | multiply-adds | vs dense | rel. error | decompose");
    println!("  -----+-------------+----------+---------------+----------+------------+----------");

    let dense_ops = rows * cols;
    for &rank in &ranks {
        if rank == 0 || rank > rows.min(cols) {
            eprintln!("skipping rank {rank}: outside 1..={}", rows.min(cols));
            continue;
        }
        let t = Instant::now();
        // LowRankError does not implement std::error::Error, and teaching it to would
        // mean editing a module this tool only means to use.
        let factors = low_rank_decompose(&dense, rows, cols, rank)
            .map_err(|e| anyhow::anyhow!("decompose at rank {rank}: {e:?}"))?;
        let took = t.elapsed();

        let reconstruction = factors.materialize();
        let metrics = rank_metrics(rows, cols, rank, &reconstruction, &dense)
            .map_err(|e| anyhow::anyhow!("metrics at rank {rank}: {e:?}"))?;

        // Arithmetic per matrix-vector product: A*(B*x) touches each factor once, while
        // a dense product touches every element. This is the half WPC cannot win, since
        // an unpacked weight still costs a full-size multiply.
        let factor_ops = rows * rank + rank * cols;

        println!(
            "  {:>4} | {:>11.2} | {:>7.1}x | {:>13} | {:>7.1}x | {:>10.4} | {:>8.1?}",
            rank,
            metrics.factor_bytes_f32 as f64 / 1048576.0,
            metrics.compression_ratio,
            factor_ops,
            dense_ops as f64 / factor_ops as f64,
            metrics.relative_frobenius_error,
            took
        );
    }

    println!();
    println!("already measured for this tensor, for comparison (gemv_2026-08-25_1522.log):");
    println!("  WPC v4 packed   :  5.31 MiB      7.5x   full-size arithmetic   1.043 ms on the card");
    println!("  dense f32       : 40.00 MiB      1.0x   full-size arithmetic     19 ms on the processor");
    println!();
    println!("gate: relative error at rank 64 below 0.10 means low-rank is a candidate;");
    println!("      above 0.25 means these weights are not low-rank and the same decomposer");
    println!("      should be pointed at a difference between two models instead.");

    Ok(())
}
