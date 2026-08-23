mod classifier;

use classifier::classify;
use std::process::{Command, Stdio};

#[derive(Clone, Copy)]
struct Gate {
    name: &'static str,
    command: &'static str,
    args: &'static [&'static str],
}

const GATES: &[Gate] = &[
    Gate { name: "format", command: "cargo", args: &["fmt", "--all", "--check"] },
    Gate { name: "build", command: "cargo", args: &["build", "--workspace", "--release"] },
    Gate { name: "test", command: "cargo", args: &["test", "--workspace", "--release"] },
    Gate { name: "clippy", command: "cargo", args: &["clippy", "-p", "wpc-runtime", "--all-targets", "--release", "--", "-D", "warnings"] },
    Gate { name: "bench-compile", command: "cargo", args: &["bench", "-p", "wpc-runtime", "--bench", "attention_bench", "--no-run"] },
];

fn main() {
    println!("AIONS Local CI — deterministic verification runner");
    for gate in GATES {
        println!("\n== {} ==", gate.name);
        let output = Command::new(gate.command)
            .args(gate.args)
            .stdin(Stdio::null())
            .output()
            .unwrap_or_else(|e| panic!("failed to start {}: {e}", gate.name));

        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{stdout}\n{stderr}");
            let diagnostic = classify(gate.name, &combined);
            eprintln!("FAIL: {}", gate.name);
            eprintln!("  kind: {:?}", diagnostic.kind);
            eprintln!("  message: {}", diagnostic.message);
            if let Some(file) = diagnostic.file { eprintln!("  file: {file}"); }
            if let Some(line) = diagnostic.line { eprintln!("  line: {line}"); }
            std::process::exit(output.status.code().unwrap_or(1));
        }
        print!("{}", String::from_utf8_lossy(&output.stdout));
        println!("PASS: {}", gate.name);
    }
    println!("\nGREEN: all local CI gates passed");
}
