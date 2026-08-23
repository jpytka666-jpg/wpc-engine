use aions_memory_kv::{CompressionExperiment, CompressionInput};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_compression_probe");
    for &run_length in &[32usize, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("rle_probe", run_length),
            &run_length,
            |b, &run_length| {
                b.iter(|| {
                    let input = CompressionInput::new("bench", 1 << 20, run_length);
                    black_box(CompressionExperiment::run(input).expect("probe"));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_probe);
criterion_main!(benches);
