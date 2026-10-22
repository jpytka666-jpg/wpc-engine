// End-to-end WPC4 compression of a *synthetic* stacked Qwen3.6 expert tensor.
// No model weights, no CUDA and no Darkstar involved. CI evidence for 3D support.
use safetensors::{tensor::{serialize_to_file, TensorView}, Dtype};
use std::{fs, process::Command, time::{SystemTime, UNIX_EPOCH}};

#[test]
fn qwen36_stacked_3d_experts_roundtrip_metadata_and_payload_size() {
    let suffix=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let root=std::env::temp_dir().join(format!("codeone-wpc-qwen36-{}-{}",std::process::id(),suffix));
    let input=root.join("input");
    let output=root.join("output");
    fs::create_dir_all(&input).unwrap();

    // 8 expert slices × 8 rows × 128 weights. Actual model uses 256 experts;
    // this miniature fixture verifies the *packing*, not model quality.
    let expert_weights=(0..(8*8*128))
        .flat_map(|i|((i%17) as f32 / 16.0).to_le_bytes())
        .collect::<Vec<_>>();
    let view=TensorView::new(Dtype::F32,vec![8,8,128],&expert_weights).unwrap();

    let vision_data=vec![0u8;2*2*8*8*4];
    let vision=TensorView::new(Dtype::F32,vec![2,2,8,8],&vision_data).unwrap();

    serialize_to_file(
      vec![
        ("model.language_model.layers.0.mlp.experts.gate_up_proj",view),
        ("model.visual.patch_embed.proj.weight",vision)
      ],
      &None,
      &input.join("mini.safetensors")
    ).unwrap();

    let res=Command::new(env!("CARGO_BIN_EXE_wpc-full-compiler"))
        .args(["--input",input.to_str().unwrap(),"--output",output.to_str().unwrap(),"--scheme","v4"])
        .output().unwrap();
    assert!(res.status.success(),"compiler failed: {}",String::from_utf8_lossy(&res.stderr));

    let meta_text=fs::read_to_string(output.join("model_v4.meta")).unwrap();
    let meta:serde_json::Value=serde_json::from_str(&meta_text).unwrap();
    let layers=meta["layers"].as_array().unwrap();
    assert_eq!(layers.len(),1,"vision 4D kernel must remain uncompressed: {}",meta_text);
    assert_eq!(layers[0]["name"],"model.language_model.layers.0.mlp.experts.gate_up_proj");
    assert_eq!(layers[0]["shape"],serde_json::json!([8,8,128]));
    let bytes=fs::metadata(output.join("model_v4.wpc")).unwrap().len();
    assert_eq!(bytes, 8192/128*68);
    assert_eq!(layers[0]["size_bytes"],bytes);

    fs::remove_dir_all(&root).unwrap();
}
