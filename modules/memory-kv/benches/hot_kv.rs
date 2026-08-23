use aions_memory_kv::HotKvBuffer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_append(c: &mut Criterion) {
    c.bench_function("hot_kv_append_128", |b| {
        b.iter(|| {
            let mut buffer = HotKvBuffer::new();
            let entries = (0..128).map(|_| vec![0u8; 64]).collect();
            buffer
                .append(0, black_box(entries))
                .expect("append");
        });
    });
}

fn bench_read(c: &mut Criterion) {
    let mut buffer = HotKvBuffer::new();
    buffer
        .append(0, (0..4096).map(|_| vec![0u8; 64]).collect())
        .expect("seed");

    c.bench_function("hot_kv_read_1024", |b| {
        b.iter(|| black_box(buffer.read(1024, 2048).expect("read")));
    });
}

criterion_group!(benches, bench_append, bench_read);
criterion_main!(benches);
