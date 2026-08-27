/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * ==========================================
 */

use noworodek::{
    ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle,
    TensorSpec, TinyTransformerConfig, WeightSetHeader, WeightSetId, WeightSetManager,
    WeightSetManifest, WeightSetVersion,
};
use noworodek::model::transformer_backprop::train_step_ce;

#[derive(Clone, Copy)]
struct Lesson {
    id: &'static str,
    language: &'static str,
    concept: &'static str,
    train: &'static str,
    heldout: &'static str,
}

const LESSONS: &[Lesson] = &[
    Lesson { id:"rust.variables.001", language:"Rust", concept:"let/mut", train:"let x = 10; let mut y = 20; y = y + x;", heldout:"let mut a = 3; a = a + 4;" },
    Lesson { id:"rust.control.002", language:"Rust", concept:"if/else", train:"if x > 0 { 1 } else { 0 }", heldout:"if x == 0 { 0 } else { 1 }" },
    Lesson { id:"rust.loops.003", language:"Rust", concept:"for", train:"for i in 0..4 { sum += i; }", heldout:"for i in 0..3 { sum += i; }" },
    Lesson { id:"rust.functions.004", language:"Rust", concept:"fn", train:"fn add(a:i32,b:i32)->i32 { a+b }", heldout:"fn mul(a:i32,b:i32)->i32 { a*b }" },
    Lesson { id:"rust.vec.005", language:"Rust", concept:"Vec", train:"let v = vec![1,2,3];", heldout:"let v = vec![4,5,6];" },
    Lesson { id:"rust.struct.006", language:"Rust", concept:"struct", train:"struct Point { x:i32, y:i32 }", heldout:"struct Size { w:i32, h:i32 }" },
    Lesson { id:"rust.enum.007", language:"Rust", concept:"enum", train:"enum Color { Red, Green, Blue }", heldout:"enum State { On, Off }" },
    Lesson { id:"rust.result.008", language:"Rust", concept:"Result", train:"fn parse()->Result<i32,String> { Ok(1) }", heldout:"fn load()->Result<i32,String> { Err(String::new()) }" },
    Lesson { id:"cpp.variables.009", language:"C++", concept:"variables", train:"int x = 10; int y = 20; y = y + x;", heldout:"int a = 3; a = a + 4;" },
    Lesson { id:"cpp.control.010", language:"C++", concept:"if/else", train:"if (x > 0) return 1; else return 0;", heldout:"if (x == 0) return 0; else return 1;" },
    Lesson { id:"cpp.loops.011", language:"C++", concept:"for", train:"for (int i=0;i<4;i++) sum += i;", heldout:"for (int i=0;i<3;i++) sum += i;" },
    Lesson { id:"cpp.functions.012", language:"C++", concept:"function", train:"int add(int a,int b){return a+b;}", heldout:"int mul(int a,int b){return a*b;}" },
    Lesson { id:"cpp.vector.013", language:"C++", concept:"vector", train:"std::vector<int> v{1,2,3};", heldout:"std::vector<int> v{4,5,6};" },
    Lesson { id:"cpp.class.014", language:"C++", concept:"class", train:"class Point { public: int x; int y; };", heldout:"class Size { public: int w; int h; };" },
    Lesson { id:"cpp.pointer.015", language:"C++", concept:"pointer", train:"int x=7; int* p=&x;", heldout:"int y=9; int* q=&y;" },
    Lesson { id:"cpp.raii.016", language:"C++", concept:"RAII", train:"std::unique_ptr<int> p(new int(7));", heldout:"std::unique_ptr<int> p(new int(9));" },
];

fn hash8(text: &str) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    (h % 8) as usize
}

fn ids(text: &str) -> Vec<usize> {
    let all: Vec<usize> = text.split_whitespace().map(hash8).collect();
    if all.is_empty() { return vec![0, 1, 2]; }
    // Keep the fixture within the decoder's bounded context while preserving the target token.
    if all.len() <= 8 { return all; }
    let mut out = all[..7].to_vec();
    out.push(*all.last().unwrap());
    out
}

fn fixture() -> (WeightSetManager, ExternalTransformer) {
    let vocab=8usize; let hidden=4usize; let intermediate=8usize;
    let specs = vec![
        TensorSpec::new("model.embeddings.token.weight", vec![vocab,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.attention.q_proj.weight", vec![hidden,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.attention.k_proj.weight", vec![hidden,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.attention.v_proj.weight", vec![hidden,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.attention.o_proj.weight", vec![hidden,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.mlp.up_proj.weight", vec![hidden,intermediate], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.mlp.gate_proj.weight", vec![hidden,intermediate], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.mlp.down_proj.weight", vec![intermediate,hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.attention_norm.weight", vec![hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.layers.00.mlp_norm.weight", vec![hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.final_norm.weight", vec![hidden], DType::F32, "school-v1"),
        TensorSpec::new("model.lm_head.weight", vec![vocab,hidden], DType::F32, "school-v1"),
    ];
    let manifest = WeightSetManifest::new(
        WeightSetHeader::new(WeightSetId::new("coder-school-v1"), WeightSetVersion::new("0.1.0").unwrap(), ArchitectureId::new("noworodek-decoder-v0"))
            .with_capabilities(["externalized-parameters","observable","editable","coder-school-v1"])
            .with_provenance("M.Szul via GPT-5.6 Luna; Noworodek Rust/C++ curriculum prototype"),
        specs,
    ).unwrap();
    let data = [
        ("model.embeddings.token.weight", vec![0.05;32]),
        ("model.layers.00.attention.q_proj.weight", vec![0.02;16]),
        ("model.layers.00.attention.k_proj.weight", vec![0.02;16]),
        ("model.layers.00.attention.v_proj.weight", vec![0.02;16]),
        ("model.layers.00.attention.o_proj.weight", vec![0.02;16]),
        ("model.layers.00.mlp.up_proj.weight", vec![0.01;32]),
        ("model.layers.00.mlp.gate_proj.weight", vec![0.01;32]),
        ("model.layers.00.mlp.down_proj.weight", vec![0.01;32]),
        ("model.layers.00.attention_norm.weight", vec![1.0;4]),
        ("model.layers.00.mlp_norm.weight", vec![1.0;4]),
        ("model.final_norm.weight", vec![1.0;4]),
        ("model.lm_head.weight", vec![0.02;32]),
    ];
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let id = manager.mount(Box::new(MemoryWeightBackend::with_tensor_data(manifest, data))).unwrap();
    let model = ExternalTransformer::new(TinyTransformerConfig{vocab_size:vocab,hidden_size:hidden,intermediate_size:intermediate,sequence_length:8,num_layers:1,rms_norm_eps:1e-5},id);
    (manager,model)
}

fn ce(model:&ExternalTransformer,manager:&WeightSetManager,tokens:&[usize],target:usize)->f32{
    let logits=model.forward(manager,tokens).unwrap();
    let v=model.config.vocab_size; let row=&logits.values()[(tokens.len()-1)*v..tokens.len()*v];
    let m=row.iter().copied().fold(f32::NEG_INFINITY,f32::max); let exps:Vec<f32>=row.iter().map(|x|(*x-m).exp()).collect(); let z:f32=exps.iter().sum(); -((exps[target]/z.max(f32::MIN_POSITIVE)).max(1e-12)).ln()
}

fn main(){
    println!("NOWORODEK CODER SCHOOL V1");
    println!("curriculum_lessons={} languages=Rust,C++ mode=curriculum+backprop+heldout", LESSONS.len());
    let (mut manager,model)=fixture();
    for lesson in LESSONS {
        let train=ids(lesson.train); let held=ids(lesson.heldout); let target=train.last().copied().unwrap_or(0);
        let before=ce(&model,&manager,&train,target);
        let mut after=before;
        for _ in 0..8 { after=train_step_ce(&model,&mut manager,&train,target,0.01).unwrap().loss_after; }
        let held_target=held.last().copied().unwrap_or(0); let held_loss=ce(&model,&manager,&held,held_target);
        println!("lesson={} language={} concept={} train_loss={:.6}->{:.6} heldout_loss={:.6} train_tokens={} heldout_tokens={}",lesson.id,lesson.language,lesson.concept,before,after,held_loss,train.len(),held.len());
    }
    let q=ParameterHandle::new(model.weight_set.clone(),"model.layers.00.attention.q_proj.weight").unwrap().read(&manager).unwrap();
    println!("FINAL q_proj_l2={:.9} q_proj_max_abs={:.9}", q.values().iter().map(|x|x*x).sum::<f32>().sqrt(), q.values().iter().fold(0.0_f32,|m,x|m.max(x.abs())));
    println!("RESULT coder_school_v1_complete=true note=hashed-8-symbol fixture; semantic code learning requires larger tokenizer/model next");
}
