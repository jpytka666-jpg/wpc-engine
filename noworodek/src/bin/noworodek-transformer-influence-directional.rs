use noworodek::{ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle, Tensor, TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};

#[derive(Debug)]
struct Score { tensor: &'static str, probes: usize, mean_delta: f32, max_delta: f32, rms_delta: f32 }

fn xs(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    ((*seed as f32) / (u32::MAX as f32)) * 2.0 - 1.0
}
fn tensor_values(len: usize, scale: f32, phase: f32) -> Vec<f32> { (0..len).map(|i| scale * (((i as f32)+phase)*0.37).sin()).collect() }

fn build() -> (WeightSetManager, ExternalTransformer, WeightSetId) {
    let vocab=8usize; let hidden=4usize; let intermediate=8usize;
    let specs=vec![
        TensorSpec::new("model.embeddings.token.weight",vec![vocab,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight",vec![hidden,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight",vec![hidden,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight",vec![hidden,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight",vec![hidden,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight",vec![hidden,intermediate],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight",vec![hidden,intermediate],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight",vec![intermediate,hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.attention_norm.weight",vec![hidden],DType::F32,"directional"),
        TensorSpec::new("model.layers.00.mlp_norm.weight",vec![hidden],DType::F32,"directional"),
        TensorSpec::new("model.final_norm.weight",vec![hidden],DType::F32,"directional"),
        TensorSpec::new("model.lm_head.weight",vec![vocab,hidden],DType::F32,"directional"),
    ];
    let names=[
        ("model.embeddings.token.weight",tensor_values(vocab*hidden,0.20,0.0)),
        ("model.layers.00.attention.q_proj.weight",tensor_values(hidden*hidden,0.30,1.0)),
        ("model.layers.00.attention.k_proj.weight",tensor_values(hidden*hidden,0.22,2.0)),
        ("model.layers.00.attention.v_proj.weight",tensor_values(hidden*hidden,0.18,3.0)),
        ("model.layers.00.attention.o_proj.weight",tensor_values(hidden*hidden,0.16,4.0)),
        ("model.layers.00.mlp.up_proj.weight",tensor_values(hidden*intermediate,0.12,5.0)),
        ("model.layers.00.mlp.gate_proj.weight",tensor_values(hidden*intermediate,0.11,6.0)),
        ("model.layers.00.mlp.down_proj.weight",tensor_values(intermediate*hidden,0.10,7.0)),
        ("model.layers.00.attention_norm.weight",vec![1.0;hidden]),
        ("model.layers.00.mlp_norm.weight",vec![1.0;hidden]),
        ("model.final_norm.weight",vec![1.0;hidden]),
        ("model.lm_head.weight",tensor_values(vocab*hidden,0.15,8.0)),
    ];
    let manifest=WeightSetManifest::new(WeightSetHeader::new(WeightSetId::new("transformer-directional-v1"),WeightSetVersion::new("0.1.0").unwrap(),ArchitectureId::new("noworodek-decoder-v0")),specs).unwrap();
    let backend=MemoryWeightBackend::with_tensor_data(manifest,names);
    let mut manager=WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let id=manager.mount(Box::new(backend)).unwrap();
    let model=ExternalTransformer::new(TinyTransformerConfig{vocab_size:vocab,hidden_size:hidden,intermediate_size:intermediate,sequence_length:8,num_layers:1,rms_norm_eps:1e-5},id.clone());
    (manager,model,id)
}

fn main(){
    let tokens=[0usize,1,2]; let delta=0.01f32; let probes=8usize;
    let tensors:[(&'static str,Vec<usize>);12]=[
        ("model.embeddings.token.weight",vec![8,4]),
        ("model.layers.00.attention.q_proj.weight",vec![4,4]),
        ("model.layers.00.attention.k_proj.weight",vec![4,4]),
        ("model.layers.00.attention.v_proj.weight",vec![4,4]),
        ("model.layers.00.attention.o_proj.weight",vec![4,4]),
        ("model.layers.00.mlp.up_proj.weight",vec![4,8]),
        ("model.layers.00.mlp.gate_proj.weight",vec![4,8]),
        ("model.layers.00.mlp.down_proj.weight",vec![8,4]),
        ("model.layers.00.attention_norm.weight",vec![4]),
        ("model.layers.00.mlp_norm.weight",vec![4]),
        ("model.final_norm.weight",vec![4]),
        ("model.lm_head.weight",vec![8,4]),
    ];
    let (mut manager,model,id)=build(); let baseline=model.forward(&manager,&tokens).unwrap().values().to_vec();
    let mut seed=0xA10A5EEDu32; let mut scores=Vec::new();
    for (name,shape) in tensors {
        let handle=ParameterHandle::new(id.clone(),name).unwrap(); let original=handle.read(&manager).unwrap().values().to_vec();
        let mut sum: f32=0.0; let mut sumsq: f32=0.0; let mut maxd: f32=0.0;
        for _ in 0..probes {
            let direction:Vec<f32>=(0..original.len()).map(|_|xs(&mut seed)).collect();
            let edited:Vec<f32>=original.iter().zip(direction.iter()).map(|(w,r)| *w + delta*r).collect();
            handle.write(&mut manager,&Tensor::from_vec(shape.clone(),edited).unwrap()).unwrap();
            let out=model.forward(&manager,&tokens).unwrap();
            let rms=(baseline.iter().zip(out.values()).map(|(a,b)|{let d=*b-*a;d*d}).sum::<f32>()/(baseline.len() as f32)).sqrt();
            sum+=rms; sumsq+=rms*rms; maxd=maxd.max(rms);
            handle.write(&mut manager,&Tensor::from_vec(shape.clone(),original.clone()).unwrap()).unwrap();
        }
        let mean=sum/(probes as f32); let rms_mean=(sumsq/(probes as f32)).sqrt();
        scores.push(Score{tensor:name,probes,mean_delta:mean,max_delta:maxd,rms_delta:rms_mean});
    }
    scores.sort_by(|a,b|b.rms_delta.total_cmp(&a.rms_delta));
    println!("NOWORODEK TRANSFORMER DIRECTIONAL INFLUENCE V1");
    println!("delta={} probes={} tensors={}",delta,probes,scores.len());
    println!("rank,tensor,probes,mean_rms,max_rms,rms_of_rms");
    for (i,s) in scores.iter().enumerate(){println!("{},{},{},{:.9},{:.9},{:.9}",i+1,s.tensor,s.probes,s.mean_delta,s.max_delta,s.rms_delta);}
}
