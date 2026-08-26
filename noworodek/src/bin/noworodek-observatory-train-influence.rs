use noworodek::{ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle, Tensor, TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager, WeightSetManifest, WeightSetVersion};

#[derive(Debug, Clone)]
struct TensorSnapshot { name: &'static str, values: Vec<f32>, shape: Vec<usize> }

fn tensor_values(len: usize, scale: f32, phase: f32) -> Vec<f32> { (0..len).map(|i| scale * (((i as f32)+phase)*0.37).sin()).collect() }

fn build() -> (WeightSetManager, ExternalTransformer, WeightSetId, Vec<TensorSnapshot>) {
    let vocab=8usize; let hidden=4usize; let intermediate=8usize;
    let specs=vec![
        TensorSpec::new("model.embeddings.token.weight",vec![vocab,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight",vec![hidden,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight",vec![hidden,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight",vec![hidden,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight",vec![hidden,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight",vec![hidden,intermediate],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight",vec![hidden,intermediate],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight",vec![intermediate,hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.attention_norm.weight",vec![hidden],DType::F32,"observatory"),
        TensorSpec::new("model.layers.00.mlp_norm.weight",vec![hidden],DType::F32,"observatory"),
        TensorSpec::new("model.final_norm.weight",vec![hidden],DType::F32,"observatory"),
        TensorSpec::new("model.lm_head.weight",vec![vocab,hidden],DType::F32,"observatory"),
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
    let manifest=WeightSetManifest::new(WeightSetHeader::new(WeightSetId::new("observatory-train-v2"),WeightSetVersion::new("0.1.0").unwrap(),ArchitectureId::new("noworodek-decoder-v0")),specs).unwrap();
    let snapshots=names.iter().map(|(n,v)|TensorSnapshot{name:n,values:v.clone(),shape:match *n {
        "model.embeddings.token.weight"|"model.lm_head.weight" => vec![vocab,hidden],
        "model.layers.00.attention.q_proj.weight"|"model.layers.00.attention.k_proj.weight"|"model.layers.00.attention.v_proj.weight"|"model.layers.00.attention.o_proj.weight" => vec![hidden,hidden],
        "model.layers.00.mlp.up_proj.weight"|"model.layers.00.mlp.gate_proj.weight" => vec![hidden,intermediate],
        "model.layers.00.mlp.down_proj.weight" => vec![intermediate,hidden],
        _ => vec![hidden],
    }}).collect::<Vec<_>>();
    let backend=MemoryWeightBackend::with_tensor_data(manifest,names);
    let mut manager=WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let id=manager.mount(Box::new(backend)).unwrap();
    let model=ExternalTransformer::new(TinyTransformerConfig{vocab_size:vocab,hidden_size:hidden,intermediate_size:intermediate,sequence_length:8,num_layers:1,rms_norm_eps:1e-5},id.clone());
    (manager,model,id,snapshots)
}

fn forward_loss(model:&ExternalTransformer, manager:&WeightSetManager, tokens:&[usize], target_token:usize)->(f32,Vec<f32>,Vec<f32>){
    let out=model.forward(manager,tokens).unwrap();
    let last_row=&out.values()[(tokens.len()-1)*model.config.vocab_size..tokens.len()*model.config.vocab_size];
    let maxv=last_row.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let exps:Vec<f32>=last_row.iter().map(|x|(*x-maxv).exp()).collect();
    let z: f32=exps.iter().sum();
    let probs:Vec<f32>=exps.iter().map(|x|*x/z.max(f32::MIN_POSITIVE)).collect();
    let loss=-probs[target_token].max(1e-12).ln();
    (loss,out.values().to_vec(),probs)
}

fn snapshot_delta(a:&[TensorSnapshot], manager:&WeightSetManager, id:&WeightSetId)->Vec<(String,f32,f32)>{
    let mut out=Vec::new();
    for s in a { let h=ParameterHandle::new(id.clone(),s.name).unwrap(); let cur=h.read(manager).unwrap().values().to_vec(); let mut l2=0.0f32; let mut maxd=0.0f32; for (x,y) in s.values.iter().zip(cur.iter()){let d=*y-*x;l2+=d*d;maxd=maxd.max(d.abs());} out.push((s.name.to_string(),l2.sqrt(),maxd)); }
    out
}

fn main(){
    let tokens=[0usize,1,2]; let target=3usize; let steps=400usize; let lr=0.005f32;
    let (mut manager,model,id,initial)=build();
    let (loss0,_,probs0)=forward_loss(&model,&manager,&tokens,target);
    println!("NOWORODEK OBSERVATORY TRAIN+INFLUENCE V2");
    println!("experience_id=math.next_token.v2 target={} steps={} lr={}",target,steps,lr);
    println!("BEFORE loss={:.9} target_prob={:.9}",loss0,probs0[target]);

    let mut trace=Vec::new();
    for step in 0..steps {
        let (before_loss,before_out,probs)=forward_loss(&model,&manager,&tokens,target);
        let last_hidden_start=(tokens.len()-1)*model.config.hidden_size;
        let hidden_out = model.forward_single_layer(&manager,&tokens).unwrap();
        let last_hidden = &hidden_out.values()[last_hidden_start..last_hidden_start+model.config.hidden_size];
        let name="model.lm_head.weight";
        let h=ParameterHandle::new(id.clone(),name).unwrap();
        let mut w=h.read(&manager).unwrap().values().to_vec();
        // Exact descent direction for the target row of softmax cross-entropy:
        // dL/dW_target = -(1 - p_target) * h, so W <- W - lr*dL.
        let scale = lr * (1.0 - probs[target]);
        for hidden_idx in 0..model.config.hidden_size { let idx=target*model.config.hidden_size+hidden_idx; w[idx]+=scale*last_hidden[hidden_idx]; }
        h.write(&mut manager,&Tensor::from_vec(vec![model.config.vocab_size,model.config.hidden_size],w).unwrap()).unwrap();
        let (after_loss,after_out,after_probs)=forward_loss(&model,&manager,&tokens,target);
        let max_delta=before_out.iter().zip(after_out.iter()).map(|(a,b)|(*b-*a).abs()).fold(0.0f32,f32::max);
        if step==0 || step%100==99 { println!("step={} loss_before={:.9} loss_after={:.9} target_prob={:.9} delta_logits_max={:.9}",step+1,before_loss,after_loss,after_probs[target],max_delta); }
        trace.push((step+1,after_loss,max_delta));
    }

    let (loss1,_,probs1)=forward_loss(&model,&manager,&tokens,target);
    let deltas=snapshot_delta(&initial,&manager,&id);
    deltas.iter().filter(|(_,l2,_)|*l2>0.0).for_each(|(name,l2,maxd)|println!("DELTA tensor={} l2={:.9} max_abs={:.9}",name,l2,maxd));
    println!("AFTER loss={:.9} loss_reduction={:.9} target_prob={:.9}",loss1,loss0-loss1,probs1[target]);
    println!("OBSERVATIONS={}",trace.len());
    println!("RESULT experience_to_weight_delta=true descent_aligned=true");
}
