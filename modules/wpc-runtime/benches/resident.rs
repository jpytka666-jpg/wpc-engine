use aions_wpc_runtime_contract::{KvPolicy, Lifecycle, ResidentSession, RuntimeLoad, Scheme, WeightsSource};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_serve_turn(c: &mut Criterion) {
    c.bench_function("resident_serve_turn", |b| {
        b.iter(|| {
            let config = RuntimeLoad {
                model_id: "qwen3-coder-30b-a3b".into(),
                scheme: Scheme::V4,
                weights_source: WeightsSource::WpcMmap,
                resident: false,
                lifecycle: Lifecycle::Cold,
                max_context: Some(32768),
                kv_policy: Some(KvPolicy::HotOnly),
            };
            let mut session = ResidentSession::load(config).expect("resident load");
            black_box(session.serve_turn().expect("serve turn"));
        });
    });
}

criterion_group!(benches, bench_serve_turn);
criterion_main!(benches);
