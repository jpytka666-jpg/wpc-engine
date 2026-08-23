use aions_memory_kv::{run_wpc_kv, WpcKvInput};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn synthetic_kv(tokens: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let mut keys = Vec::with_capacity(tokens * width);
    let mut values = Vec::with_capacity(tokens * width);
    for token in 0..tokens {
        for dim in 0..width {
            let x = (token as f32 * 0.13 + dim as f32 * 0.07).sin();
            keys.push(x);
            values.push((x * 0.83 + 0.11).cos());
        }
    }
    (keys, values)
}

fn bench_wpc_kv(c: &mut Criterion) {
    let mut group = c.benchmark_group("wpc_kv_compression");
    for &(tokens, width) in &[(64usize, 32usize), (128, 64), (256, 64)] {
        let (keys, values) = synthetic_kv(tokens, width);
        group.bench_with_input(
            BenchmarkId::new("WPC-KV", format!("{}x{}", tokens, width)),
            &(keys, values),
            |b, (keys, values)| {
                b.iter(|| {
                    black_box(
                        run_wpc_kv(WpcKvInput {
                            session_id: "bench".into(),
                            keys: keys.clone(),
                            values: values.clone(),
                            vector_width: width,
                            pattern_count: 16,
                            residual_count: 256,
                            train_iters: 5,
                        })
                        .expect("WPC KV benchmark"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_wpc_kv);
criterion_main!(benches);
