use std::fs;
use wpc_readonly_bridge::ReadonlyWpcArtifact;

#[test]
fn tensor_views_are_bounded_and_metadata_is_preserved() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("model_v3.meta"),
        r#"{"layers":[{"name":"tensor.a","shape":[2,4],"offset_bytes":0,"size_bytes":8},{"name":"tensor.b","shape":[1,4],"offset_bytes":8,"size_bytes":4}],"block_size":128}"#,
    )
    .expect("meta");
    fs::write(dir.path().join("model_v3.wpc"), 0u8..12u8).expect("payload");

    let artifact = ReadonlyWpcArtifact::open(dir.path(), "model_v3").expect("open");
    assert_eq!(artifact.tensor_info("tensor.a").unwrap().shape, vec![2, 4]);
    assert_eq!(artifact.tensor_bytes("tensor.b").unwrap(), &[8, 9, 10, 11]);
    assert_eq!(artifact.tensor_names().count(), 2);
}

#[test]
fn overlapping_ranges_are_rejected_before_exposing_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        dir.path().join("model_v3.meta"),
        r#"{"layers":[{"name":"a","shape":[1],"offset_bytes":0,"size_bytes":8},{"name":"b","shape":[1],"offset_bytes":4,"size_bytes":4}],"block_size":128}"#,
    )
    .expect("meta");
    fs::write(dir.path().join("model_v3.wpc"), [0u8; 8]).expect("payload");

    let error = ReadonlyWpcArtifact::open(dir.path(), "model_v3").expect_err("must reject overlap");
    assert!(error.to_string().contains("overlapping tensor ranges"));
}
