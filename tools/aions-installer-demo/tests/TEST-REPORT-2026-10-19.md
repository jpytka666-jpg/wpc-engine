# AIONS AI-native Installer demo — observed test results (19 October 2026)

Status: LAB PROTOTYPE PASS; NOT a complete AIONS installation.

- Rust/Cargo offline unit tests: **4/4 passed**.
- Install sequence: inspect, plan, approved install, verify, repeat/idempotence, rollback — PASS.
- Negative tests: **7/7 expected refusals** (missing consent, traversal, missing sandbox marker, modified payload, modified installed content, unexpected file, unknown manifest field).
- Public HTTPS: installer page HTTP 200; downloadable ZIP HTTP 200, 12,396 bytes.
- Public download byte-identical to original ZIP; SHA-256 `53e2d33627ab7e13edacb0ed0e09d0f791314eaecc40b2ad1651ac691c102400`.
- ZIP extracted and rebuilt offline using Cargo.lock on Fedora ZBook; repeat inspect/plan/install/verify/rollback passed.
- Existing private Government remains inaccessible via the showcase public domain.

Limits: only local Rust sandbox Markdown example, not full AIONS. Browser preview is not remote installation. No independent clean-VM, Windows/macOS or full security audit; no real LLM-based unattended install; ZIP checksum is not a digital signature.

Owner authorised creating/testing this demo. Production AIONS services/weights were not changed.
