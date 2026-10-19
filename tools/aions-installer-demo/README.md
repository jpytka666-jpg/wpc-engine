# AIONS AI-native Installer — laboratory demo v0.1.0

This is a **real but limited Rust prototype**. It creates exactly two personalised demonstration files in a marked local sandbox. It does NOT install production AIONS, AI models, system services, GPU drivers or cloud accounts. The web wizard is a browser-only preview.

## Requirements

Linux and an authorised terminal with Rust/Cargo. Read AGENTS.md and all stage prompts. Inspect source before building. Dependencies are version-locked in Cargo.lock; review them and acquire them securely. This package has no Python and the Rust executable makes no external network requests.

## Commands from the extracted package root

1. Build: cargo build --locked --release
2. Create isolated workspace: mkdir -p lab-sandbox
3. Mark it explicitly: printf 'AIONS_INSTALLER_DEMO_SANDBOX_V1\n' > lab-sandbox/.aions-demo-sandbox
4. Review personalization.example.json. It must contain exactly display_name, focus, language; do not include passwords.
5. Read-only inspection: ./target/release/aions-installer-demo inspect --package . --sandbox ./lab-sandbox --target my-lab --profile ./personalization.example.json
6. Read-only plan: repeat with plan instead of inspect.
7. After the user has specifically approved the two file changes, run install with the same options and --approve I_APPROVE_LAB_ONLY.
8. Run verify with the same options and inspect lab-sandbox/my-lab/AIONS_DEMO.md and ledger.json.
9. Test idempotence by repeating install with approval. After another explicit approval, use rollback with --approve I_APPROVE_LAB_ONLY to remove only the generated files.

## Safety boundaries

The tool refuses an unmarked sandbox, target traversal, unknown manifest fields, duplicate keys, mismatched SHA-256, missing consent, unknown profile fields, symlinks, unexpected installed files and modified ledger/content. It never invokes a shell, downloads modules or touches systemd. Full AIONS install remains future engineering work. ZIP checksum is integrity information, not a publisher signature.

Web demo: https://showcase.lore-arc.org/installer
