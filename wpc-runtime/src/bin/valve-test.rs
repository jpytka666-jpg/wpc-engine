/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 01:10:00
 * REASON FOR CREATION: The memory valve has unit tests proving it cuts a cache correctly,
 *   which proves nothing about whether a model can still hold a conversation afterwards.
 *   That is the only question worth answering before the valve is wired into anything, and
 *   it cannot be answered by a unit test: it needs the real weights and a real
 *   conversation. This runs the same conversation twice, once with the valve and once
 *   without, and asks the same recall question at the end of both.
 * MECHANICS: Four turns, then a fifth asking what was discussed at the start. With the
 *   valve on, everything the model said is dropped after each turn and replaced by a
 *   digest built ONLY from what the person actually typed. That is the policy: the user's
 *   words are trusted, the model's own are not, so a figure it invented cannot survive
 *   into the next turn. With the valve off, the raw cache grows as usual. Same weights,
 *   same questions, same decode settings; the only difference is the valve.
 * SYSTEM PART: WPC runtime, resident inference lane, verification.
 * ARCHITECTURE FUNCTION: The measurement gate for Layer 3 of AIONS_MASTER_BUILD_PLAN.md.
 *   Nothing should depend on the valve until this shows the thread survives it.
 * DEPENDENCIES/LINKS: wpc_runtime::resident::{ResidentEngine, ResidentSession}.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-multiturn-2026-08-25\wpc-runtime\src\bin\valve-test.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/resident-multi-turn
 * ==========================================
 */

use std::path::PathBuf;

use clap::Parser;
use wpc_runtime::resident::ResidentEngine;

#[derive(Parser, Debug)]
#[command(author, version, about = "Does a conversation survive having its memory cleaned?")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    wpc: PathBuf,
    #[arg(long, default_value = "v4")]
    scheme: String,
    /// Clean the model's own output out of the cache after every turn.
    #[arg(long, default_value_t = false)]
    valve: bool,
    /// What goes back into the cleaned cache: "sentences" repeats what the person typed,
    /// "facts" reduces it to bare name = value pairs.
    #[arg(long, default_value = "sentences")]
    digest: String,
    #[arg(long, default_value_t = 70)]
    max_tokens: usize,
}

/// The conversation. Deliberately carries facts the recall question depends on.
const TURNS: &[&str] = &[
    "My laptop has a Quadro M2000M graphics card with 4096 MB of memory.",
    "The model I want to run takes up 2038 MB of that.",
    "Explain in two sentences why a small model can still be useful.",
    "Name one thing that is stored in graphics memory besides the model itself.",
];

/// The same facts written as bare pairs rather than sentences.
///
/// The first run showed the model copying "M2000M" as "m2oOa" and "4096" as "4０96" even
/// when the original sentence sat right in front of it. If the failure is transcription
/// rather than memory, then a shorter thing to transcribe should survive better. Each
/// entry corresponds to the turn of the same index in TURNS.
const FACTS: &[&str] = &[
    "GPU = Quadro M2000M | VRAM = 4096 MB",
    "MODEL_SIZE = 2038 MB",
    "",
    "",
];

const RECALL: &str = "Without guessing: which graphics card did I say I have, and how much memory does it have?";

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let engine = ResidentEngine::load(&a.model, &a.wpc, &a.scheme)?;
    let mut s = engine.start_session();

    println!("=== valve: {} ===", if a.valve { "ON" } else { "OFF" });

    // Everything the person typed, and nothing the model said. With the valve on this is
    // what goes back into the cleaned cache: the model cannot corrupt a record it does
    // not write.
    let mut said_by_user: Vec<String> = Vec::new();

    for (i, q) in TURNS.iter().enumerate() {
        let (answer, cost) = s.ask(q, a.max_tokens)?;
        said_by_user.push((*q).to_string());

        let shown: String = answer.chars().take(160).collect();
        println!("\n--- turn {} ---\nQ: {q}\nA: {shown}", i + 1);
        println!(
            "   cache {} positions, of which {} is the model's own output",
            cost.cache_positions,
            s.pressure()
        );

        if a.valve {
            // The clean ground is the system prompt plus the first question, established
            // once and never re-cut.
            if i == 0 {
                s.mark_clean();
                println!("   clean mark set at {} positions", s.clean_mark());
                continue;
            }
            let digest = if a.digest == "facts" {
                // Bare pairs. In this harness they are written by hand beside the
                // questions; extracting them from a real conversation automatically is
                // the open problem, and pretending otherwise would make this measurement
                // say more than it can.
                FACTS[..=i]
                    .iter()
                    .filter(|f| !f.is_empty())
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                said_by_user
                    .iter()
                    .map(|u| format!("- the person said: {u}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let released = s.relieve(&digest)?;
            println!(
                "   VALVE: released {released} positions, cache now {}",
                s.positions()
            );
        }
    }

    println!("\n=== recall question ===\nQ: {RECALL}");
    let (answer, cost) = s.ask(RECALL, a.max_tokens)?;
    println!("A: {answer}");
    println!(
        "\ncache at end: {} positions | prefill {:?} | decode {:?}",
        cost.cache_positions, cost.prefill, cost.decode
    );

    // The two facts the recall question is about. Reported, not judged: whether the
    // answer is good enough is a human call, and printing a verdict this program is not
    // entitled to make would be worse than printing nothing.
    let lower = answer.to_lowercase();
    println!(
        "contains \"m2000m\": {} | contains \"4096\" or \"4 gb\": {}",
        lower.contains("m2000m"),
        lower.contains("4096") || lower.contains("4 gb") || lower.contains("4gb")
    );
    Ok(())
}
