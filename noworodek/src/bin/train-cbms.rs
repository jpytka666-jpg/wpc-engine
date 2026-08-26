/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 20:38:03
 * REASON FOR CREATION: Every trainer in this crate learns from data written into its own
 *   source. None reads a corpus. So the one question that decides whether CBMS can carry
 *   a model - can a model predict the next CBMS symbol better than chance - could not be
 *   asked. This asks it.
 * MECHANICS: Reads a flat little-endian u16 file of ids produced by `cbms ids`, holds the
 *   last tenth back, and trains next-id prediction with the crate's existing analytic
 *   backprop. Reports training loss against held-out loss and against the loss of pure
 *   chance, ln(vocab). Training loss alone proves memorisation; only the held-out figure
 *   distinguishes learning from it, which is why both are printed side by side and why
 *   the held-out tenth is never sampled for training.
 * SYSTEM PART: Noworodek, CBMS training lane.
 * ARCHITECTURE FUNCTION: The first measurement of whether a CBMS vocabulary is learnable
 *   at all. Everything larger is gated behind its answer.
 * DEPENDENCIES/LINKS: noworodek::model::transformer_backprop::train_step_ce,
 *   ParameterRegistry::register_decoder_transformer, MemoryWeightBackend.
 *   Input produced by cbms-writing's `ids` command.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-cbms-train\noworodek\src\bin\train-cbms.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek/cbms-training
 * ==========================================
 */

use noworodek::model::transformer_backprop::train_step_ce;
use noworodek::{
    ArchitectureId, DType, ExternalTransformer, MemoryWeightBackend, ParameterHandle,
    ParameterRegistry, Tensor,
    TinyTransformerConfig, WeightSetId, WeightSetManager, WeightSetVersion,
};
use std::process::ExitCode;

/// Deterministic, so a result can be repeated. A measurement that cannot be reproduced
/// is not a measurement.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn read_ids(path: &str) -> std::io::Result<Vec<usize>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) as usize)
        .collect())
}

/// Cross entropy of the last position, forward only. Used for the held-out figure, which
/// must never touch the training path.
fn loss_at(
    model: &ExternalTransformer,
    mgr: &WeightSetManager,
    tokens: &[usize],
    target: usize,
) -> Option<f32> {
    let logits = model.forward(mgr, tokens).ok()?;
    let v = model.config.vocab_size;
    let row = &logits.values()[(tokens.len() - 1) * v..tokens.len() * v];
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = row.iter().map(|x| (*x - m).exp()).sum();
    Some(-((row[target] - m).exp() / z.max(f32::MIN_POSITIVE)).ln())
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// Weights on disk: name length, name, value count, values. Nothing clever - a format a
/// person can read with a hex editor is a format that can be debugged at three in the
/// morning, and a daemon that runs overnight will eventually need that.
///
/// Without this the trainer learns in memory and exits, so every nightly cycle would
/// start from nothing and there would be no earlier state to roll back to.
fn save_weights(path: &str, names: &[String], mgr: &WeightSetManager, set: &WeightSetId) -> std::io::Result<usize> {
    use std::io::Write;
    let mut out = Vec::new();
    let mut written = 0usize;
    for name in names {
        let Ok(handle) = ParameterHandle::new(set.clone(), name) else { continue };
        let Ok(tensor) = handle.read(mgr) else { continue };
        let values = tensor.values();
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        written += 1;
    }
    std::fs::File::create(path)?.write_all(&out)?;
    Ok(written)
}

fn load_weights(path: &str, mgr: &mut WeightSetManager, set: &WeightSetId) -> std::io::Result<usize> {
    let bytes = std::fs::read(path)?;
    let mut at = 0usize;
    let mut restored = 0usize;
    while at + 4 <= bytes.len() {
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        if at + n > bytes.len() { break }
        let name = String::from_utf8_lossy(&bytes[at..at + n]).to_string();
        at += n;
        if at + 8 > bytes.len() { break }
        let count = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
        at += 8;
        if at + count * 4 > bytes.len() { break }
        let values: Vec<f32> = bytes[at..at + count * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        at += count * 4;
        // Shape comes from the tensor already mounted, so the file carries values only.
        // A shape saved twice is a shape that can disagree with itself.
        if let Ok(handle) = ParameterHandle::new(set.clone(), &name) {
            if let Ok(current) = handle.read(mgr) {
                if current.values().len() == values.len() {
                    if let Ok(t) = Tensor::from_vec(current.shape().to_vec(), values) {
                        if handle.write(mgr, &t).is_ok() {
                            restored += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(restored)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(ids_path) = arg(&args, "--ids") else {
        eprintln!(
            "train-cbms --ids <plik.u16> --vocab <n> [--hidden 256] [--inter 512] \
             [--layers 1] [--seq 8] [--steps 2000] [--lr 0.01] [--seed 7]"
        );
        return ExitCode::from(2);
    };
    let num = |n: &str, d: usize| arg(&args, n).and_then(|s| s.parse().ok()).unwrap_or(d);
    let vocab = num("--vocab", 0);
    let hidden = num("--hidden", 256);
    let inter = num("--inter", 512);
    let layers = num("--layers", 1);
    let seq = num("--seq", 8);
    let steps = num("--steps", 2000);
    let seed = num("--seed", 7) as u64;
    let lr: f32 = arg(&args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(0.01);

    let ids = match read_ids(&ids_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read {ids_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    if ids.len() < seq * 20 {
        eprintln!("corpus too short: {} ids", ids.len());
        return ExitCode::FAILURE;
    }
    let highest = ids.iter().copied().max().unwrap_or(0);
    if vocab == 0 || highest >= vocab {
        eprintln!("--vocab must exceed the highest id in the file ({highest})");
        return ExitCode::FAILURE;
    }

    // The last tenth is never sampled for training. Without it, a falling loss says only
    // that the model memorised what it was shown.
    let split = ids.len() * 9 / 10;
    let (train, held) = ids.split_at(split);

    let mut registry = ParameterRegistry::new();
    if let Err(e) = registry.register_decoder_transformer(layers, vocab, hidden, inter, DType::F32)
    {
        eprintln!("cannot register model: {e:?}");
        return ExitCode::FAILURE;
    }
    let manifest = match registry.to_manifest(
        WeightSetId::new("cbms-training-v1"),
        WeightSetVersion::new("0.1.0").unwrap(),
        ArchitectureId::new("noworodek-decoder-v0"),
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot build manifest: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    // Small random start. Norm weights begin at one: a norm scaled to zero would kill the
    // signal before the first step could say anything.
    let mut rng = Rng(seed | 1);
    let data: Vec<(String, Vec<f32>)> = manifest
        .tensors()
        .iter()
        .map(|spec| {
            let n: usize = spec.shape.iter().product();
            let values = if spec.name.ends_with("norm.weight") {
                vec![1.0; n]
            } else {
                let scale = (2.0 / (spec.shape.first().copied().unwrap_or(1) as f32)).sqrt();
                (0..n).map(|_| rng.unit() * scale).collect()
            };
            (spec.name.clone(), values)
        })
        .collect();
    let total_params: usize = data.iter().map(|(_, v)| v.len()).sum();

    // `with_tensor_data` wants names that live forever, because every caller before this
    // one passed string literals. This model builds its names from the layer count, so
    // they are leaked instead - a fixed few dozen strings in a program that exits. The
    // alternative was widening a shared function to suit one new caller, which is the
    // kind of change that should be asked for rather than slipped in.
    let named: Vec<(&'static str, Vec<f32>)> = data
        .into_iter()
        .map(|(n, v)| (&*Box::leak(n.into_boxed_str()), v))
        .collect();

    let mut mgr = WeightSetManager::new(ArchitectureId::new("noworodek-decoder-v0"));
    let set = match mgr.mount(Box::new(MemoryWeightBackend::with_tensor_data(
        manifest, named,
    ))) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("cannot mount weights: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let model = ExternalTransformer::new(
        TinyTransformerConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: inter,
            sequence_length: seq,
            num_layers: layers,
            rms_norm_eps: 1e-5,
        },
        set,
    );

    // Two baselines, because one of them is too easy to beat.
    //
    // Uniform chance assumes every symbol is equally likely. They are not, and a model
    // that learned NOTHING except which symbols are common would beat it comfortably
    // while understanding nothing at all. Reporting only that number would flatter any
    // result into looking like learning.
    //
    // The unigram baseline is what "knowing the frequencies and nothing else" scores.
    // Beating uniform chance is the floor; beating unigram is the first honest claim,
    // because it cannot be reached without using the context.
    let chance = (vocab as f32).ln();

    // A fixed held-out sample, drawn once, so the held-out figure at step 1 and at step
    // 2000 measure the same thing.
    let mut hrng = Rng(seed ^ 0xA5A5);
    let held_windows: Vec<(Vec<usize>, usize)> = (0..24)
        .filter_map(|_| {
            if held.len() < seq + 1 {
                return None;
            }
            let at = hrng.below(held.len() - seq - 1);
            Some((held[at..at + seq].to_vec(), held[at + seq]))
        })
        .collect();

    // What "knowing the frequencies and nothing else" scores on the SAME held-out sample.
    // Add-one smoothing, so a symbol the training half never showed costs a lot rather
    // than infinity.
    let unigram = {
        let mut counts = vec![0usize; vocab];
        for &id in train {
            counts[id] += 1;
        }
        let n = train.len() as f32;
        let mut total = 0.0;
        for (_, tgt) in &held_windows {
            let p = (counts[*tgt] as f32 + 1.0) / (n + vocab as f32);
            total += -p.ln();
        }
        if held_windows.is_empty() { f32::NAN } else { total / held_windows.len() as f32 }
    };

    let held_loss = |mgr: &WeightSetManager| -> f32 {
        let mut total = 0.0;
        let mut n = 0;
        for (ctx, tgt) in &held_windows {
            if let Some(l) = loss_at(&model, mgr, ctx, *tgt) {
                total += l;
                n += 1;
            }
        }
        if n == 0 { f32::NAN } else { total / n as f32 }
    };

    println!("NOWORODEK CBMS TRAINING V1");
    println!("korpus         : {} numerow ({} uczace, {} odlozone)", ids.len(), train.len(), held.len());
    println!("slownik        : {vocab} (najwyzszy uzyty {highest})");
    println!("model          : hidden {hidden}, inter {inter}, warstw {layers}, sekwencja {seq}");
    println!("parametrow     : {total_params}");
    println!("krokow         : {steps}, lr {lr}, ziarno {seed}");
    println!();
    println!("PROG 1, kostka : {chance:.4}   <- wszystkie symbole rownie prawdopodobne");
    println!("PROG 2, czestosc: {unigram:.4}   <- SAMA znajomosc czestosci, bez kontekstu");
    println!("                            bicie progu 2 to pierwsza uczciwa teza");
    println!("odlozone PRZED : {:.4}", held_loss(&mgr));
    println!();

    // Continue from an earlier cycle if asked. Without this every nightly run would start
    // from nothing, and there would be no earlier state to return to.
    if let Some(from) = arg(&args, "--load") {
        match load_weights(&from, &mut mgr, &model.weight_set) {
            Ok(n) => println!("wczytano       : {n} tensorow z {from}"),
            Err(e) => {
                eprintln!("nie moge wczytac {from}: {e}");
                return ExitCode::FAILURE;
            }
        }
        println!("odlozone po wczytaniu: {:.4}", held_loss(&mgr));
        println!();
    }

    let start = std::time::Instant::now();
    let mut window = 0.0f32;
    let mut window_n = 0usize;
    let report_every = (steps / 20).max(1);

    // Stop when the held-out figure stops improving. Measured on this corpus, held-out
    // flattened around step 5000 while training loss kept falling - that gap is the model
    // memorising, and every step past it costs time and makes the result worse, not
    // better. `--patience 0` disables it for a deliberate long run.
    let patience = num("--patience", 4);
    // Start from what this cycle was HANDED, not from nothing. Beginning at infinity let
    // the first improvement inside a cycle count as "best" even when it was worse than
    // the state loaded from the previous cycle - so each night handed on something a
    // little worse while every individual run looked like it had improved. Measured:
    // cycle 2 loaded 5.8937 and saved 6.1721 before this line existed.
    let mut best_held = held_loss(&mgr);
    // The state handed in counts as the best so far. Without this a cycle that never
    // improved had nothing to restore and saved whatever the last step produced -
    // measured: cycle 3 loaded 5.1904 and would have saved 5.9704.
    let mut since_best = 0usize;
    let mut stopped_early = None;
    let mut best_weights: Vec<(String, Vec<f32>)> = registry
        .iter()
        .filter_map(|p| {
            let h = ParameterHandle::new(model.weight_set.clone(), &p.name).ok()?;
            Some((p.name.clone(), h.read(&mgr).ok()?.values().to_vec()))
        })
        .collect();

    for step in 1..=steps {
        let at = rng.below(train.len() - seq - 1);
        let ctx = &train[at..at + seq];
        let tgt = train[at + seq];
        match train_step_ce(&model, &mut mgr, ctx, tgt, lr) {
            Ok(r) => {
                window += r.loss_after;
                window_n += 1;
            }
            Err(e) => {
                eprintln!("krok {step} nie powiodl sie: {e:?}");
                return ExitCode::FAILURE;
            }
        }
        if step % report_every == 0 {
            let tr = window / window_n.max(1) as f32;
            let hl = held_loss(&mgr);
            println!(
                "krok {step:>6} | uczace {tr:>8.4} | odlozone {hl:>8.4} | {:>6.0} ms/krok",
                start.elapsed().as_millis() as f32 / step as f32
            );
            window = 0.0;
            window_n = 0;

            // A real improvement, not noise in the third decimal.
            if hl < best_held - 0.01 {
                best_held = hl;
                since_best = 0;
                // Keep the weights that earned this figure. Early stopping without this
                // saves whatever the last step happened to produce, which is by
                // definition worse than the best - and over many nightly cycles that
                // drifts steadily downhill while every individual run looks fine.
                best_weights = registry
                    .iter()
                    .filter_map(|p| {
                        let h = ParameterHandle::new(model.weight_set.clone(), &p.name).ok()?;
                        Some((p.name.clone(), h.read(&mgr).ok()?.values().to_vec()))
                    })
                    .collect();
            } else {
                since_best += 1;
                if patience > 0 && since_best >= patience {
                    stopped_early = Some(step);
                    println!();
                    println!("STOP: odlozone nie poprawilo sie przez {patience} pomiarow.");
                    println!("      Najlepsze bylo {best_held:.4}. Dalsze kroki tylko");
                    println!("      pogleboja zapamietywanie - to nie jest awaria, to koniec nauki");
                    println!("      z TEGO materialu przy TEJ wielkosci modelu.");
                    break;
                }
            }
        }
    }

    let mut final_held = held_loss(&mgr);

    if let Some(to) = arg(&args, "--save") {
        // Put the best state back before saving, so a cycle never hands its successor
        // something worse than what it actually achieved.
        if !best_weights.is_empty() {
            let mut restored = 0usize;
            for (name, values) in &best_weights {
                if let Ok(h) = ParameterHandle::new(model.weight_set.clone(), name) {
                    if let Ok(cur) = h.read(&mgr) {
                        if let Ok(t) = Tensor::from_vec(cur.shape().to_vec(), values.clone()) {
                            if h.write(&mut mgr, &t).is_ok() { restored += 1; }
                        }
                    }
                }
            }
            println!("przywrocono    : najlepszy stan ({restored} tensorow, odlozone {best_held:.4})");
        }
        // Report what was actually saved. Reading the figure before the restore made the
        // summary understate its own result - it printed the last step, not the best.
        final_held = held_loss(&mgr);
        let names: Vec<String> = registry.iter().map(|p| p.name.clone()).collect();
        match save_weights(&to, &names, &mgr, &model.weight_set) {
            Ok(n) => println!("zapisano       : {n} tensorow do {to}"),
            Err(e) => eprintln!("nie moge zapisac {to}: {e}"),
        }
    }

    println!();
    if let Some(at) = stopped_early {
        println!("zatrzymany na kroku {at} z {steps} - odlozone stanelo");
    }
    println!("PROG LOSOWY    : {chance:.4}");
    println!("odlozone PO    : {final_held:.4}");
    println!("czas calkowity : {:.1} s", start.elapsed().as_secs_f32());
    println!();
    if final_held.is_nan() {
        println!("WYNIK: nie da sie orzec - odlozona czesc nie dala sie policzyc");
        return ExitCode::FAILURE;
    }
    let bits = |nats: f32| nats / std::f32::consts::LN_2;
    if final_held >= chance {
        println!("WYNIK: NIC. Nie bije nawet rzutu kostka.");
        println!("       To nie dowodzi, ze CBMS sie nie da nauczyc - dowodzi, ze TA");
        println!("       konfiguracja przy tylu krokach tego nie zrobila.");
    } else if final_held >= unigram {
        println!("WYNIK: UCZY SIE, ALE PLYTKO.");
        println!("       Bije kostke o {:.3} bitu na symbol,", bits(chance - final_held));
        println!("       ale wciaz jest o {:.3} bitu GORSZY niz sama znajomosc czestosci.", bits(final_held - unigram));
        println!("       Czyli nie uzywa jeszcze kontekstu - uczy sie dopiero, co jest czeste.");
    } else {
        println!("WYNIK: UZYWA KONTEKSTU.");
        println!("       Bije sama znajomosc czestosci o {:.3} bitu na symbol,", bits(unigram - final_held));
        println!("       a rzut kostka o {:.3} bitu.", bits(chance - final_held));
        println!("       Tego nie da sie osiagnac bez patrzenia na poprzednie symbole.");
    }
    ExitCode::SUCCESS
}
