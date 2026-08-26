/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 22:17:38
 * REASON FOR CREATION: A held-out loss says a model learned SOMETHING. It cannot say what.
 *   The only honest answer to "what did it learn from watching me" is to give it a context
 *   and read what it expects to come next.
 * MECHANICS: Loads a checkpoint, feeds contexts taken from the corpus itself, and prints
 *   the symbols it ranks highest - decoded through the code book back into Polish, because
 *   a list of numbers answers nothing. Also reports how much of its probability mass sits
 *   on its single favourite symbol overall, which is how a model that has learned one
 *   boilerplate phrase gives itself away.
 * SYSTEM PART: Noworodek, CBMS training lane.
 * ARCHITECTURE FUNCTION: The instrument that turns a loss figure into a statement about
 *   content. Without it, "it is learning" is a number nobody can argue with.
 * DEPENDENCIES/LINKS: reads a checkpoint written by train-cbms and an id file from
 *   `cbms ids`; symbol names resolved by the caller against the same book.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-cbms-train\noworodek\src\bin\probe-cbms.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
 * ==========================================
 */

use noworodek::{
    ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle,
    ParameterRegistry, Tensor, TinyTransformerConfig, WeightSetId, WeightSetManager,
    WeightSetVersion,
};
use std::process::ExitCode;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(ids_path), Some(weights)) = (arg(&args, "--ids"), arg(&args, "--load")) else {
        eprintln!("probe-cbms --ids <plik.u16> --load <wagi.bin> --vocab <n> \
                   [--hidden 64] [--inter 128] [--seq 8] [--top 8] [--prob 6]");
        return ExitCode::from(2);
    };
    let num = |n: &str, d: usize| arg(&args, n).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (vocab, hidden, inter, seq) =
        (num("--vocab", 0), num("--hidden", 64), num("--inter", 128), num("--seq", 8));
    let top = num("--top", 8);
    let probes = num("--prob", 6);

    let bytes = match std::fs::read(&ids_path) {
        Ok(b) => b,
        Err(e) => { eprintln!("nie moge przeczytac {ids_path}: {e}"); return ExitCode::FAILURE }
    };
    let ids: Vec<usize> = bytes.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) as usize).collect();
    if ids.len() < seq + 2 { eprintln!("za malo numerow"); return ExitCode::FAILURE }

    let mut registry = ParameterRegistry::new();
    if registry.register_decoder_transformer(1, vocab, hidden, inter, DType::F32).is_err() {
        eprintln!("nie moge zlozyc modelu"); return ExitCode::FAILURE
    }
    let manifest = match registry.to_manifest(
        WeightSetId::new("probe"), WeightSetVersion::new("0.1.0").unwrap(),
        ArchitectureId::new("noworodek-decoder-v0")) {
        Ok(m) => m, Err(e) => { eprintln!("{e:?}"); return ExitCode::FAILURE }
    };
    let zeros: Vec<(&'static str, Vec<f32>)> = manifest.tensors().iter()
        .map(|s| {
            let n: usize = s.shape.iter().product();
            (&*Box::leak(s.name.clone().into_boxed_str()), vec![0.01f32; n])
        }).collect();
    let mut mgr = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let set = match mgr.mount(Box::new(MemoryWeightBackend::with_tensor_data(manifest, zeros))) {
        Ok(s) => s, Err(e) => { eprintln!("{e:?}"); return ExitCode::FAILURE }
    };

    // Load the checkpoint. The same complete-or-refuse rules as the trainer.
    let raw = match std::fs::read(&weights) {
        Ok(b) => b, Err(e) => { eprintln!("{weights}: {e}"); return ExitCode::FAILURE }
    };
    if raw.len() < 12 || &raw[..4] != b"NWRD" {
        eprintln!("to nie jest plik wag Noworodka"); return ExitCode::FAILURE
    }
    let mut at = 12usize;
    let mut loaded = 0usize;
    while at + 4 <= raw.len() {
        let n = u32::from_le_bytes(raw[at..at+4].try_into().unwrap()) as usize; at += 4;
        if at + n > raw.len() { break }
        let name = String::from_utf8_lossy(&raw[at..at+n]).to_string(); at += n;
        if at + 8 > raw.len() { break }
        let count = u64::from_le_bytes(raw[at..at+8].try_into().unwrap()) as usize; at += 8;
        if at + count*4 > raw.len() { break }
        let vals: Vec<f32> = raw[at..at+count*4].chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        at += count*4;
        if let Ok(h) = ParameterHandle::new(set.clone(), &name) {
            if let Ok(cur) = h.read(&mgr) {
                if let Ok(t) = Tensor::from_vec(cur.shape().to_vec(), vals) {
                    if h.write(&mut mgr, &t).is_ok() { loaded += 1 }
                }
            }
        }
    }
    println!("wczytano {loaded} tensorow z {weights}");

    let model = ExternalTransformer::new(TinyTransformerConfig {
        vocab_size: vocab, hidden_size: hidden, intermediate_size: inter,
        sequence_length: seq, num_layers: 1, rms_norm_eps: 1e-5 }, set);

    // How concentrated is it? A model that learned one boilerplate phrase puts most of
    // its confidence on the same few symbols no matter what it was shown - which looks
    // exactly like learning when only the loss is reported.
    let mut favourite = vec![0usize; vocab];
    let step = (ids.len() / 200).max(1);
    let mut sampled = 0usize;
    let mut at_pos = 0usize;
    while at_pos + seq + 1 < ids.len() && sampled < 200 {
        if let Ok(logits) = model.forward(&mgr, &ids[at_pos..at_pos+seq]) {
            let row = &logits.values()[(seq-1)*vocab..seq*vocab];
            let best = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            favourite[best] += 1;
            sampled += 1;
        }
        at_pos += step;
    }
    let (fav_id, fav_n) = favourite.iter().enumerate()
        .max_by_key(|(_, n)| **n).map(|(i, n)| (i, *n)).unwrap_or((0, 0));
    println!();
    println!("na {sampled} roznych kontekstach:");
    println!("  najczestsza pierwsza odpowiedz : symbol {fav_id}, {fav_n} razy \
              ({:.0}% wszystkich)", 100.0 * fav_n as f32 / sampled.max(1) as f32);
    println!("  roznych pierwszych odpowiedzi  : {}",
             favourite.iter().filter(|n| **n > 0).count());
    println!("  <- malo roznych = przewiduje szablon, nie tresc");

    println!();
    for k in 0..probes {
        let start = (ids.len() / (probes + 1)) * (k + 1);
        if start + seq + 1 >= ids.len() { break }
        let ctx = &ids[start..start+seq];
        let truth = ids[start+seq];
        let Ok(logits) = model.forward(&mgr, ctx) else { continue };
        let row = &logits.values()[(seq-1)*vocab..seq*vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = row.iter().map(|x| (*x - m).exp()).sum();
        let mut ranked: Vec<(usize, f32)> = row.iter().enumerate()
            .map(|(i, v)| (i, (*v - m).exp() / z)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rank_of_truth = ranked.iter().position(|(i, _)| *i == truth).unwrap_or(vocab);
        print!("kontekst {:?} -> prawda {truth} (miejsce {}), typy:",
               &ctx[seq.saturating_sub(4)..], rank_of_truth + 1);
        for (id, p) in ranked.iter().take(top) {
            print!(" {id}:{:.0}%", p * 100.0);
        }
        println!();
    }
    ExitCode::SUCCESS
}
