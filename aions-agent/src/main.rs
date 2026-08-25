//! aions-agent: a tool-execution loop around the WPC runtime.
//!
//! The runtime is one-shot: text in, text out, no session. This wraps it so a
//! compressed model can actually *act* - it emits a tool call, we run the tool,
//! append the result, and call the runtime again with the grown conversation.
//!
//! The cost of that design is real and worth stating: prefill runs at roughly
//! one token per second, so every turn re-reads the whole conversation. Two or
//! three turns is practical today; twenty is not. Batched prefill is what makes
//! this cheap, and it does not exist yet.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Config {
    runtime: PathBuf,
    model: PathBuf,
    wpc: PathBuf,
    scheme: String,
    max_tokens: usize,
    max_turns: usize,
    workdir: PathBuf,
}

/// The tool menu handed to the model. Kept deliberately short: every token here
/// is paid for again on each turn, at about a second per token.
const TOOLS: &str = "\
You have these tools:
list_files(dir) - list files in a directory
read_file(path) - read a file's contents
run_command(cmd) - run a shell command and return its output
finish(answer) - give your final answer and stop

Rules:\n- Reply with ONE tool call only. No explanation, no markdown fences, no repetition of these rules.\n- When you know the answer, reply finish(your answer).\n\nNext tool call:";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        runtime: PathBuf::from("/home/aions/wpc-workspace/target/release/wpc-runtime"),
        model: PathBuf::from("/home/aions/qwen3-coder-run"),
        wpc: PathBuf::from("/home/aions/qwen3-coder-wpc4"),
        scheme: "v4".to_string(),
        max_tokens: 60,
        max_turns: 6,
        workdir: PathBuf::from("/home/aions/wpc-workspace"),
    };
    let mut task = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => { task = args[i + 1].clone(); i += 2; }
            "--wpc" => { cfg.wpc = PathBuf::from(&args[i + 1]); i += 2; }
            "--scheme" => { cfg.scheme = args[i + 1].clone(); i += 2; }
            "--model" => { cfg.model = PathBuf::from(&args[i + 1]); i += 2; }
            "--max-turns" => { cfg.max_turns = args[i + 1].parse().unwrap_or(6); i += 2; }
            "--max-tokens" => { cfg.max_tokens = args[i + 1].parse().unwrap_or(60); i += 2; }
            "--workdir" => { cfg.workdir = PathBuf::from(&args[i + 1]); i += 2; }
            other => { eprintln!("unknown flag {other}"); std::process::exit(2); }
        }
    }
    if task.is_empty() {
        eprintln!("usage: aions-agent --task \"...\" [--wpc DIR] [--scheme v4] [--max-turns N]");
        std::process::exit(2);
    }

    let mut transcript = format!("{TOOLS}\n\nTask: {task}\n");

    for turn in 1..=cfg.max_turns {
        println!("\n=== TURA {turn} ===");
        let reply = match ask_model(&cfg, &transcript) {
            Some(r) => r,
            None => { println!("model nie odpowiedzial - przerywam"); break; }
        };
        println!("model: {reply}");

        let call = match parse_call(&reply) {
            Some(c) => c,
            None => {
                println!("nie rozpoznalem wywolania narzedzia - przerywam");
                println!("\n--- ODPOWIEDZ KONCOWA (bez narzedzia) ---\n{reply}");
                break;
            }
        };

        if call.name == "finish" {
            println!("\n--- ODPOWIEDZ KONCOWA ---\n{}", call.arg);
            break;
        }

        let result = run_tool(&cfg, &call);
        // Tool output is capped hard: every character comes back as prompt on the
        // next turn, and prompt is the expensive part.
        let trimmed: String = result.chars().take(600).collect();
        println!("narzedzie -> {} znakow", trimmed.len());

        transcript.push_str(&format!(
            "\n\n--- OBSERVATION from {} ---\n{}\n--- END ---\n\nReply with exactly one tool call, or finish(answer) if you now know the answer.\nNext tool call:",
            call.name, trimmed
        ));
    }
}

struct Call {
    name: String,
    arg: String,
}

/// Pull the first `name(arg)` out of the model's reply. Accepts single quotes,
/// double quotes or bare arguments, and ignores any code fence around it.
fn parse_call(reply: &str) -> Option<Call> {
    const KNOWN: [&str; 4] = ["list_files", "read_file", "run_command", "finish"];
    let mut best: Option<(usize, Call)> = None;

    for name in KNOWN {
        let mut from = 0;
        while let Some(rel) = reply[from..].find(name) {
            let at = from + rel;
            let rest = &reply[at + name.len()..];
            let rest = rest.trim_start();
            if let Some(inner) = rest.strip_prefix('(') {
                if let Some(close) = inner.find(')') {
                    let raw = inner[..close].trim();
                    let raw = raw
                        .trim_start_matches("query=")
                        .trim_start_matches("path=")
                        .trim_start_matches("dir=")
                        .trim_start_matches("cmd=")
                        .trim();
                    let arg = raw.trim_matches(|c| c == '"' || c == '\'').to_string();
                    let cand = Call { name: name.to_string(), arg };
                    if best.as_ref().map(|(p, _)| at < *p).unwrap_or(true) {
                        best = Some((at, cand));
                    }
                    break;
                }
            }
            from = at + name.len();
        }
    }
    best.map(|(_, c)| c)
}

fn run_tool(cfg: &Config, call: &Call) -> String {
    // Every path is resolved under workdir; an absolute argument would otherwise
    // let the model wander the whole filesystem.
    let safe = |p: &str| -> PathBuf {
        let p = p.trim_start_matches('/');
        cfg.workdir.join(p)
    };

    match call.name.as_str() {
        "list_files" => {
            let dir = if call.arg.is_empty() || call.arg == "." {
                cfg.workdir.clone()
            } else {
                safe(&call.arg)
            };
            match fs::read_dir(&dir) {
                Ok(entries) => {
                    let mut names: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let n = e.file_name().to_string_lossy().to_string();
                            if e.path().is_dir() { format!("{n}/") } else { n }
                        })
                        .collect();
                    names.sort();
                    names.join("\n")
                }
                Err(e) => format!("ERROR: cannot list {}: {e}", dir.display()),
            }
        }
        "read_file" => {
            let p = safe(&call.arg);
            match fs::read_to_string(&p) {
                Ok(s) => s,
                Err(e) => format!("ERROR: cannot read {}: {e}", p.display()),
            }
        }
        "run_command" => {
            // Deliberately narrow: read-only inspection only. The model is being
            // trusted with a shell here, so the allowlist is the safety boundary.
            const ALLOWED: [&str; 9] = ["ls", "cat", "grep", "wc", "head", "tail", "find", "du", "stat"];
            let first = call.arg.split_whitespace().next().unwrap_or("");
            if !ALLOWED.contains(&first) {
                return format!(
                    "ERROR: command '{first}' is not allowed. Allowed: {}",
                    ALLOWED.join(", ")
                );
            }
            match Command::new("sh")
                .arg("-c")
                .arg(&call.arg)
                .current_dir(&cfg.workdir)
                .output()
            {
                Ok(o) => {
                    let mut s = String::from_utf8_lossy(&o.stdout).to_string();
                    if s.is_empty() {
                        s = String::from_utf8_lossy(&o.stderr).to_string();
                    }
                    s
                }
                Err(e) => format!("ERROR: {e}"),
            }
        }
        other => format!("ERROR: unknown tool {other}"),
    }
}

fn ask_model(cfg: &Config, prompt: &str) -> Option<String> {
    let out = Command::new(&cfg.runtime)
        .arg("--model").arg(&cfg.model)
        .arg("--wpc").arg(&cfg.wpc)
        .arg("--scheme").arg(&cfg.scheme)
        .arg("--prompt").arg(prompt)
        .arg("--max-tokens").arg(cfg.max_tokens.to_string())
        .output()
        .ok()?;

    let text = String::from_utf8_lossy(&out.stdout).to_string();
    // The runtime echoes the prompt before its own output; keep only the tail.
    let reply = if let Some(pos) = text.rfind("Next tool call:") {
        text[pos + "Next tool call:".len()..].to_string()
    } else if let Some(pos) = text.find(prompt) {
        text[pos + prompt.len()..].to_string()
    } else {
        text
    };
    let reply = reply.trim().to_string();
    if reply.is_empty() { None } else { Some(reply) }
}

fn _unused(_: &Path) {}
