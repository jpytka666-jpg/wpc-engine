use wpc_runtime::resident::{ResidentRuntime, ResidentState};

#[test]
fn resident_runtime_starts_cold_and_transitions_to_resident() {
    let mut runtime = ResidentRuntime::new("qwen3-moe-wpc-v4");
    assert_eq!(runtime.state(), ResidentState::Cold);

    runtime.load().expect("resident load should succeed");
    assert_eq!(runtime.state(), ResidentState::Resident);
    assert_eq!(runtime.model_id(), "qwen3-moe-wpc-v4");
}

#[test]
fn resident_runtime_load_is_idempotent() {
    let mut runtime = ResidentRuntime::new("qwen3-moe-wpc-v4");
    runtime.load().expect("first load should succeed");
    runtime.load().expect("second load should be harmless");
    assert_eq!(runtime.state(), ResidentState::Resident);
}
