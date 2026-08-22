// benches/attention_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::prelude::*;
use wpc_runtime::forward_batch::{BatchEngine, KvLayer, MmapF32};

fn generate_random_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..rows*cols).map(|_| rng.gen::<f32>()).collect()
}

fn get_rss_kb() -> Option<u64> {
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(res_pages) = parts[1].parse::<u64>() {
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
                return Some(res_pages * page_size / 1024);
            }
        }
    }
    None
}

fn bench_attention(c: &mut Criterion) {
    let dim = 128usize; // smaller for quick benches
    let batch_vals = [1usize, 2, 4, 8];
    let seq_base = 64usize;

    for &b in &batch_vals {
        let s = seq_base + b;
        let q = generate_random_matrix(b, dim);
        let k = generate_random_matrix(s, dim);
        let v = generate_random_matrix(s, dim);
        let engine = BatchEngine::new(dim);

        let mut kv = KvLayer::with_capacity(dim, s).expect("kv alloc");
        kv.append_batch(&k, &v, s).expect("append");

        c.bench_with_input(BenchmarkId::new("optimized_gemm", b), &b, |bencher, &_b| {
            bencher.iter(|| {
                let _ = engine.optimized_attention_batch(&q, kv.keys_ptr(), kv.vals_ptr(), b, s, seq_base).unwrap();
            });
        });

        // reference implementation
        let q_vecs: Vec<Vec<f32>> = (0..b).map(|i| q[i*dim..(i+1)*dim].to_vec()).collect();
        c.bench_with_input(BenchmarkId::new("reference", b), &b, |bencher, &_b| {
            bencher.iter(|| {
                let _ = engine.reference_attention_batch(&q_vecs, &kv);
            });
        });

        // memory snapshot
        if let Some(rss) = get_rss_kb() {
            println!("bench b={} rss_kb={}", b, rss);
        }
    }
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
