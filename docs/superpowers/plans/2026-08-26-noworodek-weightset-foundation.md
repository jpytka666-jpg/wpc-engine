# Noworodek WeightSet Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first independently testable Noworodek subsystem: a model-agnostic, versioned WeightSet contract and lifecycle manager that can later back a from-scratch operator model and WPC without coupling storage into Transformer code.

**Architecture:** Introduce a focused Rust crate for WeightSet contracts and lifecycle management rather than modifying existing WPC tensor code. A `WeightSetManager` owns mounted sets, validates compatibility before activation, and exposes backend-neutral metadata; initial storage is an in-memory test backend, with the interface designed so mmap and WPC can be added without changing the model-facing contract.

**Tech Stack:** Rust workspace, Cargo, serde/serde_json only if already established by workspace conventions, standard Rust testing. No new ML framework is introduced in this phase.

**Spec:** `docs/superpowers/specs/2026-08-26-noworodek-design.md`

## Global Constraints

- The model core and parameter storage remain separate.
- WeightSets are first-class and independently addressable.
- A WeightSet exposes stable identity, version, architecture compatibility, tensor manifest, checksums, provenance, and capability metadata.
- The manager rejects incompatible sets rather than silently applying them.
- WPC is a later backend; do not embed WPC assumptions into the contract.
- Every milestone requires executable tests and evidence.
- Do not alter the existing WPC runtime behavior in this foundation phase.

---

### Task 1: Create the Noworodek WeightSet crate boundary

**Files:**
- Create: `noworodek/Cargo.toml`
- Create: `noworodek/src/lib.rs`
- Modify: `Cargo.toml`
- Test: `noworodek/src/lib.rs` module tests

**Interfaces:**
- Consumes: existing workspace Cargo conventions.
- Produces: a `noworodek` library crate exporting the WeightSet subsystem without changing existing WPC crates.

- [ ] **Step 1: Write the failing workspace membership test/check**

Add the crate to the workspace and add a minimal unit test proving the crate compiles and its public module is reachable:

```rust
#[test]
fn noworodek_crate_is_reachable() {
    assert_eq!(noworodek::VERSION, "0.1.0");
}
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cargo test -p noworodek noworodek_crate_is_reachable
```

Expected: FAIL because the crate does not yet exist.

- [ ] **Step 3: Add the minimal crate**

Create `noworodek/Cargo.toml` with package name `noworodek`, version `0.1.0`, edition matching the workspace, and a library target. Export:

```rust
pub const VERSION: &str = "0.1.0";
pub mod weightset;
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
cargo test -p noworodek noworodek_crate_is_reachable
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml noworodek/
git commit -m "feat(noworodek): add WeightSet crate boundary"
```

### Task 2: Define the immutable WeightSet manifest contract

**Files:**
- Create: `noworodek/src/weightset.rs`
- Test: `noworodek/src/weightset.rs` unit tests

**Interfaces:**
- Consumes: Task 1 crate boundary.
- Produces: `WeightSetId`, `WeightSetVersion`, `ArchitectureId`, `TensorSpec`, `WeightSetManifest`, `WeightSetState`, and `WeightSetError`.

- [ ] **Step 1: Write failing manifest tests**

The tests must establish these exact behaviors:

```rust
#[test]
fn manifest_preserves_identity_version_and_compatibility() {
    let manifest = test_manifest("coding", "1.0.0");
    assert_eq!(manifest.name.as_str(), "coding");
    assert_eq!(manifest.version.to_string(), "1.0.0");
    assert_eq!(manifest.architecture.as_str(), "noworodek-v0");
}

#[test]
fn manifest_contains_tensor_shape_and_dtype() {
    let manifest = test_manifest("coding", "1.0.0");
    let tensor = manifest.tensor("core.layers.0.attn.q").unwrap();
    assert_eq!(tensor.shape, vec![8, 8]);
    assert_eq!(tensor.dtype, DType::F32);
}

#[test]
fn duplicate_tensor_names_are_rejected() {
    let result = WeightSetManifest::new(test_header(), vec![tensor("x"), tensor("x")]);
    assert!(matches!(result, Err(WeightSetError::DuplicateTensor(_))));
}
```

Use a small `DType` enum sufficient for the contract (`F32`, `F16`, `BF16`, `I8`) and a tensor checksum field. Do not implement actual tensor storage in this task.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cargo test -p noworodek weightset
```

Expected: FAIL because the contract types are not implemented.

- [ ] **Step 3: Implement the contract**

Implement typed IDs/version wrappers, tensor metadata, manifest construction/validation, lookup by tensor name, declared capabilities, provenance string, and overall manifest checksum. Keep the manifest immutable after construction.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
cargo test -p noworodek weightset
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add noworodek/src/weightset.rs
git commit -m "feat(noworodek): define WeightSet manifest contract"
```

### Task 3: Implement backend-neutral WeightSet lifecycle

**Files:**
- Create: `noworodek/src/backend.rs`
- Modify: `noworodek/src/lib.rs`
- Test: `noworodek/src/backend.rs` unit tests

**Interfaces:**
- Consumes: `WeightSetManifest` from Task 2.
- Produces: `WeightBackend`, `MemoryWeightBackend`, `MountedWeightSet`, and `WeightSetManager`.

The core interface is:

```rust
pub trait WeightBackend {
    fn manifest(&self) -> &WeightSetManifest;
    fn load(&mut self) -> Result<(), WeightSetError>;
    fn unload(&mut self) -> Result<(), WeightSetError>;
    fn is_loaded(&self) -> bool;
}
```

The manager API is:

```rust
pub struct WeightSetManager { /* private state */ }

impl WeightSetManager {
    pub fn new(architecture: ArchitectureId) -> Self;
    pub fn mount(&mut self, backend: Box<dyn WeightBackend>) -> Result<WeightSetId, WeightSetError>;
    pub fn unmount(&mut self, id: &WeightSetId) -> Result<(), WeightSetError>;
    pub fn active(&self, id: &WeightSetId) -> Option<&MountedWeightSet>;
}
```

- [ ] **Step 1: Write failing lifecycle tests**

Tests must prove mount, unload, replacement, and architecture rejection:

```rust
#[test]
fn manager_mounts_and_unmounts_a_compatible_set() {
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
    let backend = MemoryWeightBackend::from_manifest(test_manifest("coding", "1.0.0"));
    let id = manager.mount(Box::new(backend)).unwrap();
    assert!(manager.active(&id).unwrap().is_loaded());
    manager.unmount(&id).unwrap();
    assert!(!manager.active(&id).unwrap().is_loaded());
}

#[test]
fn manager_rejects_incompatible_architecture() {
    let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
    let backend = MemoryWeightBackend::from_manifest(test_manifest_for_arch("coding", "1.0.0", "other-model"));
    let result = manager.mount(Box::new(backend));
    assert!(matches!(result, Err(WeightSetError::IncompatibleArchitecture { .. })));
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p noworodek backend
```

Expected: FAIL because lifecycle types do not yet exist.

- [ ] **Step 3: Implement minimal backend and manager**

`MemoryWeightBackend` stores only lifecycle state and manifest in this task; it does not yet pretend to contain real model tensors. `WeightSetManager::mount` validates architecture before loading. Duplicate IDs must be rejected. `unmount` must leave the set addressable for inspection but not loaded.

- [ ] **Step 4: Add replacement semantics**

Add:

```rust
pub fn replace(&mut self, id: &WeightSetId, backend: Box<dyn WeightBackend>) -> Result<(), WeightSetError>;
```

The manager must validate the replacement before unloading the existing backend. If validation fails, the existing set remains loaded and unchanged.

- [ ] **Step 5: Run focused and workspace tests**

Run:

```bash
cargo test -p noworodek
cargo test --workspace
```

Expected: PASS with no regression in existing WPC crates.

- [ ] **Step 6: Commit**

```bash
git add noworodek/src/backend.rs noworodek/src/lib.rs
 git commit -m "feat(noworodek): add WeightSet lifecycle manager"
```

### Task 4: Add hot-swap transaction safety and snapshots

**Files:**
- Modify: `noworodek/src/backend.rs`
- Test: `noworodek/src/backend.rs`
- Create: `noworodek/src/snapshot.rs`
- Modify: `noworodek/src/lib.rs`

**Interfaces:**
- Consumes: lifecycle manager from Task 3.
- Produces: validated replacement transactions and `WeightSetSnapshot` metadata.

- [ ] **Step 1: Write failing transaction tests**

Test that a failed replacement does not unload the current set, and that a snapshot records the active set identity/version/checksum.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cargo test -p noworodek snapshot
```

Expected: FAIL.

- [ ] **Step 3: Implement atomic replacement**

Validate the incoming manifest, prepare the incoming backend, then switch ownership. If preparation fails, retain the old backend untouched. If the switch succeeds, unload the old backend only after the new backend is active.

- [ ] **Step 4: Implement snapshot metadata**

A snapshot contains architecture ID, active WeightSet IDs, versions, manifest checksums, and a schema version. It does not serialize actual tensors yet.

- [ ] **Step 5: Run all tests**

Run:

```bash
cargo test -p noworodek
cargo test --workspace
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add noworodek/
git commit -m "feat(noworodek): make WeightSet swaps transactional"
```

### Task 5: Record the first verification checkpoint

**Files:**
- Create: `docs/noworodek/checkpoints/001-weightset-foundation.md`

**Interfaces:**
- Consumes: passing crate/workspace tests from Tasks 1–4.
- Produces: human-readable evidence tied to commit SHAs and exact commands.

- [ ] **Step 1: Run the verification suite**

Run:

```bash
cargo fmt --all -- --check
cargo test -p noworodek
cargo test --workspace
```

Expected: all commands PASS.

- [ ] **Step 2: Record evidence**

Write the actual test results, crate version, commit SHAs, and supported lifecycle behaviors. Do not claim tensor materialisation or WPC support; this checkpoint proves only the modular WeightSet contract and safe lifecycle mechanism.

- [ ] **Step 3: Commit the checkpoint**

```bash
git add docs/noworodek/checkpoints/001-weightset-foundation.md
git commit -m "docs(noworodek): record WeightSet foundation checkpoint"
```

## Self-review

- Spec coverage: WeightSet identity/version/compatibility, tensor manifest, checksums, provenance, capabilities, lifecycle, incompatibility rejection, hot swap, snapshots, and backend separation are covered by Tasks 2–5.
- Placeholder scan: no TBD/TODO implementation steps are used.
- Type consistency: Task 2 owns manifest types; Task 3 consumes them; Task 4 consumes the manager; Task 5 consumes the verified crate.
- Scope: this plan intentionally covers only the first independently testable subsystem. Transformer/training/AIONS environment/WPC are separate subsequent plans so each produces a working checkpoint.
