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
        let status = Command::new(gate.command)
            .args(gate.args)
            .stdin(Stdio::null())
            .status()
            .unwrap_or_else(|e| panic!("failed to start {}: {e}", gate.name));
        if !status.success() {
            eprintln!("FAIL: {}", gate.name);
            std::process::exit(status.code().unwrap_or(1));
        }
        println!("PASS: {}", gate.name);
    }
    println!("\nGREEN: all local CI gates passed");
}
