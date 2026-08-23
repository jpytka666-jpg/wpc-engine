use std::fs;
use std::sync::Arc;
use wpc_runtime::wpc_weights_v4::{WpcLinearV4, WpcModelDataV4};

#[test]
fn wpc_v4_model_data_is_shared_across_linear_layers() {
    let dir = std::env::temp_dir().join(format!(
        "wpc-resident-weight-sharing-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("create temp model directory");

    fs::write(
        dir.join("model_v4.meta"),
        r#"{
            "layers":[{"name":"test.weight","shape":[1,128],"offset_bytes":0,"size_bytes":68}],
            "block_size":128
        }"#,
    )
    .expect("write model metadata");
    fs::write(dir.join("model_v4.wpc"), vec![0u8; 68]).expect("write model data");

    let data = WpcModelDataV4::open(&dir).expect("open WPC model");
    assert_eq!(Arc::strong_count(&data), 1);

    let _layer_a = WpcLinearV4::new(data.clone(), "test.weight", 1, 128, None);
    assert_eq!(Arc::strong_count(&data), 2);

    let _layer_b = WpcLinearV4::new(data.clone(), "test.weight", 1, 128, None);
    assert_eq!(Arc::strong_count(&data), 3);

    fs::remove_dir_all(&dir).expect("remove temp model directory");
}
