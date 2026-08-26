use std::collections::BTreeMap;

use crate::{ParameterHandle, Tensor, WeightSetError, WeightSetId, WeightSetManager};
use super::transformer::{ExternalTransformer, TinyTransformerConfig};

#[derive(Clone, Debug)]
pub struct TransformerTrainReport {
    pub loss_before: f32,
    pub loss_after: f32,
    pub gradient_l2: f32,
    pub changed_tensors: usize,
}

#[derive(Clone)]
struct LayerCache {
    input: Tensor,
    norm1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn: Tensor,
    probs: Vec<Vec<f32>>,
    projected: Tensor,
    residual: Tensor,
    norm2: Tensor,
    gate: Tensor,
    up: Tensor,
    silu_gate: Tensor,
    mlp: Tensor,
    output: Tensor,
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    o_w: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    norm1_w: Tensor,
    norm2_w: Tensor,
}

struct ForwardCache {
    tokens: Vec<usize>,
    embedding: Tensor,
    layers: Vec<LayerCache>,
    final_norm: Tensor,
    final_norm_w: Tensor,
    lm_head: Tensor,
    logits: Tensor,
}

fn read(handle: &ParameterHandle, manager: &WeightSetManager) -> Result<Tensor, WeightSetError> { handle.read(manager) }

fn weight(model: &ExternalTransformer, manager: &WeightSetManager, name: String) -> Result<Tensor, WeightSetError> {
    read(&ParameterHandle::new(model.weight_set.clone(), name)?, manager)
}

fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, WeightSetError> { a.matmul(b) }

fn transpose(x: &Tensor) -> Result<Tensor, WeightSetError> {
    let (r,c) = (x.shape()[0], x.shape()[1]);
    let mut out = vec![0.0; r*c];
    for i in 0..r { for j in 0..c { out[j*r+i] = x.values()[i*c+j]; } }
    Tensor::from_vec(vec![c,r], out)
}

fn add_inplace(dst: &mut [f32], src: &[f32]) { for (a,b) in dst.iter_mut().zip(src) { *a += *b; } }

fn norm_forward(input: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, WeightSetError> {
    let seq=input.shape()[0]; let h=input.shape()[1]; let mut out=vec![0.0; seq*h];
    for i in 0..seq {
        let off=i*h; let mut ms=0.0;
        for d in 0..h { let x=input.values()[off+d]; ms += x*x; }
        let inv=(ms/(h as f32)+eps).sqrt().recip();
        for d in 0..h { out[off+d]=input.values()[off+d]*inv*w.values()[d]; }
    }
    Tensor::from_vec(vec![seq,h],out)
}

fn norm_backward(input:&Tensor, w:&Tensor, grad:&Tensor, eps:f32) -> Result<(Tensor,Tensor),WeightSetError> {
    let seq=input.shape()[0]; let h=input.shape()[1]; let mut dx=vec![0.0;seq*h]; let mut dw=vec![0.0;h];
    for i in 0..seq {
        let off=i*h; let mut ms=0.0;
        for d in 0..h { let x=input.values()[off+d]; ms += x*x; }
        let inv=(ms/(h as f32)+eps).sqrt().recip();
        let mut dot=0.0;
        for d in 0..h { dot += grad.values()[off+d]*w.values()[d]*input.values()[off+d]; dw[d]+=grad.values()[off+d]*input.values()[off+d]*inv; }
        let coeff=inv*inv*dot/(h as f32);
        for d in 0..h { dx[off+d]=w.values()[d]*inv*(grad.values()[off+d]-input.values()[off+d]*coeff); }
    }
    Ok((Tensor::from_vec(vec![seq,h],dx)?,Tensor::from_vec(vec![h],dw)?))
}

fn silu(x:f32)->f32 { x/(1.0+(-x).exp()) }
fn silu_grad(x:f32)->f32 { let s=1.0/(1.0+(-x).exp()); s + x*s*(1.0-s) }

fn attention_forward(q:&Tensor,k:&Tensor,v:&Tensor)->Result<(Tensor,Vec<Vec<f32>>),WeightSetError>{
    let seq=q.shape()[0]; let h=q.shape()[1]; let scale=(h as f32).sqrt(); let mut out=vec![0.0;seq*h]; let mut probs=Vec::with_capacity(seq);
    for i in 0..seq {
        let mut scores=Vec::with_capacity(i+1);
        for j in 0..=i { let mut dot=0.0; for d in 0..h { dot+=q.values()[i*h+d]*k.values()[j*h+d]; } scores.push(dot/scale); }
        let maxv=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);
        let mut p:Vec<f32>=scores.iter().map(|s|(s-maxv).exp()).collect(); let z:p32=0.0;
        let _=z;
        let sum: f32=p.iter().sum(); for x in &mut p { *x/=sum.max(f32::MIN_POSITIVE); }
        for (j,pj) in p.iter().enumerate() { for d in 0..h { out[i*h+d]+=*pj*v.values()[j*h+d]; } }
        probs.push(p);
    }
    Ok((Tensor::from_vec(vec![seq,h],out)?,probs))
}

fn attention_backward(q:&Tensor,k:&Tensor,v:&Tensor,probs:&[Vec<f32>],grad_out:&Tensor)->Result<(Tensor,Tensor,Tensor),WeightSetError>{
    let seq=q.shape()[0]; let h=q.shape()[1]; let scale=(h as f32).sqrt(); let mut dq=vec![0.0;seq*h]; let mut dk=vec![0.0;seq*h]; let mut dv=vec![0.0;seq*h];
    for i in 0..seq {
        let p=&probs[i]; let mut dscore=vec![0.0;p.len()];
        for j in 0..p.len() {
            let mut dp=0.0; for d in 0..h { dp += grad_out.values()[i*h+d]*v.values()[j*h+d]; dv[j*h+d]+=p[j]*grad_out.values()[i*h+d]; }
            dscore[j]=p[j]*dp;
        }
        let dot:p32=0.0; let _=dot;
        let mut sum=0.0; for j in 0..p.len() { sum += dscore[j]; }
        for j in 0..p.len() {
            let ds=p[j]*(dscore[j]-sum);
            for d in 0..h { dq[i*h+d]+=ds*k.values()[j*h+d]/scale; dk[j*h+d]+=ds*q.values()[i*h+d]/scale; }
        }
    }
    Ok((Tensor::from_vec(vec![seq,h],dq)?,Tensor::from_vec(vec![seq,h],dk)?,Tensor::from_vec(vec![seq,h],dv)?))
}

fn softmax_ce(logits:&Tensor,target:usize)->Result<(f32,Tensor),WeightSetError>{
    let seq=logits.shape()[0]; let v=logits.shape()[1]; let row=&logits.values()[(seq-1)*v..seq*v];
    let maxv=row.iter().copied().fold(f32::NEG_INFINITY,f32::max); let mut exps=Vec::with_capacity(v); for &x in row { exps.push((x-maxv).exp()); }
    let z: f32=exps.iter().sum(); let mut dlog=vec![0.0;seq*v]; let ptarget=(exps[target]/z.max(f32::MIN_POSITIVE)).max(1e-12); let loss=-ptarget.ln();
    for j in 0..v { dlog[(seq-1)*v+j]=exps[j]/z.max(f32::MIN_POSITIVE); } dlog[(seq-1)*v+target]-=1.0;
    Ok((loss,Tensor::from_vec(vec![seq,v],dlog)?))
}

fn forward_cache(model:&ExternalTransformer, manager:&WeightSetManager, tokens:&[usize])->Result<ForwardCache,WeightSetError>{
    let embedding=model.embedding(manager,tokens)?; let mut hidden=embedding.clone(); let mut layers=Vec::with_capacity(model.config.num_layers);
    for layer in 0..model.config.num_layers {
        let p=format!("model.layers.{layer:02}");
        let norm1_w=weight(model,manager,format!("{p}.attention_norm.weight"))?; let norm1=norm_forward(&hidden,&norm1_w,model.config.rms_norm_eps)?;
        let q_w=weight(model,manager,format!("{p}.attention.q_proj.weight"))?; let k_w=weight(model,manager,format!("{p}.attention.k_proj.weight"))?; let v_w=weight(model,manager,format!("{p}.attention.v_proj.weight"))?; let o_w=weight(model,manager,format!("{p}.attention.o_proj.weight"))?;
        let q=matmul(&norm1,&q_w)?; let k=matmul(&norm1,&k_w)?; let v=matmul(&norm1,&v_w)?; let (attn,probs)=attention_forward(&q,&k,&v)?; let projected=matmul(&attn,&o_w)?; let residual=hidden.add(&projected)?;
        let norm2_w=weight(model,manager,format!("{p}.mlp_norm.weight"))?; let norm2=norm_forward(&residual,&norm2_w,model.config.rms_norm_eps)?;
        let gate_w=weight(model,manager,format!("{p}.mlp.gate_proj.weight"))?; let up_w=weight(model,manager,format!("{p}.mlp.up_proj.weight"))?; let down_w=weight(model,manager,format!("{p}.mlp.down_proj.weight"))?;
        let gate=matmul(&norm2,&gate_w)?; let up=matmul(&norm2,&up_w)?; let sg=Tensor::from_vec(gate.shape().to_vec(),gate.values().iter().map(|&x|silu(x)).collect())?; let mlp=sg.hadamard(&up)?; let down=matmul(&mlp,&down_w)?; let output=residual.add(&down)?;
        layers.push(LayerCache{input:hidden.clone(),norm1,q,k,v,attn,probs,projected,residual:residual.clone(),norm2,gate,up,silu_gate:sg,mlp,output:output.clone(),q_w,k_w,v_w,o_w,gate_w,up_w,down_w,norm1_w,norm2_w}); hidden=output;
    }
    let final_norm_w=weight(model,manager,"model.final_norm.weight".into())?; let final_norm=norm_forward(&hidden,&final_norm_w,model.config.rms_norm_eps)?; let lm_head=weight(model,manager,"model.lm_head.weight".into())?; let logits=matmul(&final_norm,&transpose(&lm_head)?)?;
    Ok(ForwardCache{tokens:tokens.to_vec(),embedding,layers,final_norm,final_norm_w,lm_head,logits})
}

fn add_grad(map:&mut BTreeMap<String,Vec<f32>>,name:&str,g:&Tensor){ let slot=map.entry(name.into()).or_insert_with(||vec![0.0;g.values().len()]); add_inplace(slot,g.values()); }

fn matmul_backward(x:&Tensor,w:&Tensor,dy:&Tensor)->Result<(Tensor,Tensor),WeightSetError>{
    let wt=transpose(w)?; let dx=matmul(dy,&wt)?; let xt=transpose(x)?; let dw=matmul(&xt,dy)?; Ok((dx,dw))
}

pub fn train_step_ce(model:&ExternalTransformer, manager:&mut WeightSetManager, tokens:&[usize], target:usize, lr:f32)->Result<TransformerTrainReport,WeightSetError>{
    if tokens.is_empty() || target>=model.config.vocab_size { return Err(WeightSetError::Backend("invalid CE training input".into())); }
    let before=forward_cache(model,manager,tokens)?; let (loss,dlogits)=softmax_ce(&before.logits,target)?;
    let (d_final_norm,d_lm)=matmul_backward(&before.final_norm,&before.lm_head,&dlogits)?; let mut grads:BTreeMap<String,Vec<f32>>=BTreeMap::new();
    add_grad(&mut grads,"model.lm_head.weight",&d_lm); let (mut d_hidden, d_final_w)=norm_backward(&before.layers.last().map(|l|&l.output).cloned().unwrap_or(before.embedding.clone()),&before.final_norm_w,&d_final_norm,model.config.rms_norm_eps)?;
    add_grad(&mut grads,"model.final_norm.weight",&d_final_w);
    for layer in before.layers.iter().rev() {
        let (d_mlp, d_down)=matmul_backward(&layer.mlp,&layer.down_w,&d_hidden)?; add_grad(&mut grads, &format!("model.layers.00.mlp.down_proj.weight"), &d_down);
        let mut d_sg=vec![0.0;layer.mlp.values().len()]; let mut d_up=vec![0.0;layer.mlp.values().len()];
        for i in 0..d_sg.len(){ d_sg[i]=d_mlp.values()[i]*layer.up.values()[i]; d_up[i]=d_mlp.values()[i]*layer.silu_gate.values()[i]; }
        let d_gate=Tensor::from_vec(layer.gate.shape().to_vec(),d_sg.into_iter().enumerate().map(|(i,g)|g*silu_grad(layer.gate.values()[i])).collect())?;
        let d_up=Tensor::from_vec(layer.up.shape().to_vec(),d_up)?; let (d_norm2a,d_gate_w)=matmul_backward(&layer.norm2,&layer.gate_w,&d_gate)?; let (d_norm2b,d_up_w)=matmul_backward(&layer.norm2,&layer.up_w,&d_up)?; add_grad(&mut grads,"model.layers.00.mlp.gate_proj.weight",&d_gate_w); add_grad(&mut grads,"model.layers.00.mlp.up_proj.weight",&d_up_w);
        let mut d_norm2=vec![0.0;d_norm2a.values().len()]; for i in 0..d_norm2.len(){d_norm2[i]=d_norm2a.values()[i]+d_norm2b.values()[i];}
        let d_norm2=Tensor::from_vec(layer.norm2.shape().to_vec(),d_norm2)?; let (d_res_norm2,d_norm2_w)=norm_backward(&layer.residual,&layer.norm2_w,&d_norm2,model.config.rms_norm_eps)?; add_grad(&mut grads,"model.layers.00.mlp_norm.weight",&d_norm2_w);
        let d_residual=d_hidden.add(&d_res_norm2)?; let (d_attn,d_o_w)=matmul_backward(&layer.attn,&layer.o_w,&d_residual)?; add_grad(&mut grads,"model.layers.00.attention.o_proj.weight",&d_o_w);
        let (d_q,d_k,d_v)=attention_backward(&layer.q,&layer.k,&layer.v,&layer.probs,&d_attn)?; let (d_nq,d_q_w)=matmul_backward(&layer.norm1,&layer.q_w,&d_q)?; let (d_nk,d_k_w)=matmul_backward(&layer.norm1,&layer.k_w,&d_k)?; let (d_nv,d_v_w)=matmul_backward(&layer.norm1,&layer.v_w,&d_v)?; add_grad(&mut grads,"model.layers.00.attention.q_proj.weight",&d_q_w); add_grad(&mut grads,"model.layers.00.attention.k_proj.weight",&d_k_w); add_grad(&mut grads,"model.layers.00.attention.v_proj.weight",&d_v_w);
        let mut d_norm1=vec![0.0;d_nq.values().len()]; for i in 0..d_norm1.len(){d_norm1[i]=d_nq.values()[i]+d_nk.values()[i]+d_nv.values()[i];}
        let d_norm1=Tensor::from_vec(layer.norm1.shape().to_vec(),d_norm1)?; let (d_input,d_norm1_w)=norm_backward(&layer.input,&layer.norm1_w,&d_norm1,model.config.rms_norm_eps)?; add_grad(&mut grads,"model.layers.00.attention_norm.weight",&d_norm1_w);
        d_hidden=d_residual.add(&d_input)?;
    }
    let mut embed_grad=vec![0.0;before.embedding.values().len()]; let h=model.config.hidden_size; for (row,&tok) in before.tokens.iter().enumerate(){ for d in 0..h { embed_grad[tok*h+d]+=d_hidden.values()[row*h+d]; } }
    add_grad(&mut grads,"model.embeddings.token.weight",&Tensor::from_vec(vec![model.config.vocab_size,h],embed_grad)?);
    let mut grad_l2=0.0; let mut changed=0;
    for (name,g) in &grads { grad_l2+=g.iter().map(|x|x*x).sum::<f32>(); if g.iter().any(|x|x.abs()>1e-12){changed+=1;} let handle=ParameterHandle::new(model.weight_set.clone(),name.clone())?; let mut w=handle.read(manager)?; for (p,gg) in w.values_mut().iter_mut().zip(g){*p-=lr*gg;} handle.write(manager,&w)?; }
    let after=forward_cache(model,manager,tokens)?; let (loss_after,_)=softmax_ce(&after.logits,target)?;
    Ok(TransformerTrainReport{loss_before:loss,loss_after,gradient_l2:grad_l2.sqrt(),changed_tensors:changed})
}

#[allow(dead_code)]
fn _assert_config_used(_cfg:&TinyTransformerConfig,_id:&WeightSetId) {}
