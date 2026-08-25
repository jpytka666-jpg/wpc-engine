# AIONS Studio Implementation Plan

## Goal

Build the first Linux-first AIONS Studio vertical slice on `feature/gui` without touching Cloud/KV branches. The slice establishes the dynamic workspace and surface protocol before adding external integrations.

## Phase 1 — Core contract

1. Add `studio-core` Rust crate to the workspace.
2. Define typed `Surface`, `SurfaceType`, `SurfaceState`, `Workspace`, and presentation commands.
3. Implement validation for surface IDs, lifecycle transitions, and bounded layout hints.
4. Write unit tests before implementation for valid/invalid transitions.

## Phase 2 — Tauri shell

1. Add `studio-ui` React + TypeScript SPA.
2. Add Tauri 2 Rust shell under `studio-ui/src-tauri`.
3. Keep Tauri core thin; it hosts the workspace state and bridges future AIONS services.
4. Do not add Windows packaging or Windows CI.

## Phase 3 — Dynamic renderer

1. Implement a single workspace canvas.
2. Implement a generic `SurfaceHost`.
3. Implement initial surfaces: Agent, Code, Graph, Terminal.
4. Implement materialise/move/resize/focus/collapse/close transitions.
5. Implement dark graphite / emerald / amber visual tokens.

## Phase 4 — Protocol bridge

1. Define serializable presentation messages.
2. Bridge Rust workspace events to React.
3. Bridge user surface actions back to Rust.
4. Add capability checks around privileged commands.

## Phase 5 — Verification

1. Rust unit tests for workspace state machine.
2. TypeScript typecheck.
3. Frontend build.
4. Tauri Linux build.
5. CI records all results.

## Phase 6 — Later integrations

Only after the vertical slice is stable:

- Monaco
- terminal/PTY
- graph renderer
- browser/media
- voice
- GitHub/MCP
- Qwen 4B routing
- CBMS/memory/KV/WPC visualisation

## Definition of done for the first slice

The app launches on Linux, starts as a sparse workspace, accepts typed presentation commands, dynamically creates and transforms surfaces, renders the AIONS visual identity, and passes all core/type/build checks. No existing Cloud/KV branch is modified.
