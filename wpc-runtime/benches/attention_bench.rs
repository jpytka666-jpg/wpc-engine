use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wpc_runtime::forward_batch::{BatchEngine, KvLayer};

fn generate_random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn get_rss_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages = text.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
    Some(pages * page_size / 1024)
}

fn bench_attention(c: &mut Criterion) {
    let dim = 128usize;
    let batch_vals = [1usize, 2, 4, 8];
    let seq_base = 64usize;

    for &b in &batch_vals {
        let s = seq_base + b;
        let q = generate_random_matrix(b, dim, 0x1000 + b as u64);
        let k = generate_random_matrix(s, dim, 0x2000 + b as u64);
        let v = generate_random_matrix(s, dim, 0x3000 + b as u64);
        let engine = BatchEngine::new(dim);

        let mut kv = KvLayer::with_capacity(dim, s).expect("kv alloc");
        kv.append_batch(&k, &v, s).expect("append");

        c.bench_with_input(BenchmarkId::new("optimized_gemm", b), &b, |bencher, &_b| {
            bencher.iter(|| {
                black_box(
                    engine
                        .optimized_attention_from_kv(&q, &kv, b, seq_base)
                        .unwrap(),
                );
            });
        });

        let q_vecs: Vec<Vec<f32>> = (0..b).map(|i| q[i * dim..(i + 1) * dim].to_vec()).collect();
        c.bench_with_input(BenchmarkId::new("reference", b), &b, |bencher, &_b| {
            bencher.iter(|| {
                black_box(engine.reference_attention_batch(&q_vecs, &kv).unwrap());
            });
        });

        if let Some(rss) = get_rss_kb() {
            println!("bench b={b} rss_kb={rss}");
        }
    }
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
