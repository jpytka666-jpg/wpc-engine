# AIONS Studio — GUI Architecture & Design Specification

**Author:** M. Szul via GPT-5.6 Luna
**Date:** 2026-08-25
**Repository:** `jpytka666-jpg/wpc-engine`
**Branch:** `feature/gui`
**Status:** Design baseline

## 1. Product definition

AIONS Studio is not a conventional IDE and not a collection of permanently docked application panels. It is a **dynamic AI-directed workspace**: one primary visual surface in which the agent materializes temporary, interactive surfaces according to the user's current task.

The user primarily interacts through voice/conversation. The UI is the visual extension of AIONS: when the agent needs to explain code, architecture, email, CI, a graph, browser content, media, logs, or system state, it creates or transforms a surface. When the task is finished, surfaces can collapse, disappear, or return to a clean workspace.

The core UX principle is:

> **AIONS has a space, not a fixed set of applications.**

## 2. Platform policy

- **Linux-first and Linux-native in product intent.**
- Windows is explicitly not a target for the initial product design, CI, packaging, or UX decisions.
- macOS may be considered later as a secondary target without changing the core architecture.
- Desktop integration must remain possible through Rust/Tauri rather than through platform-specific UI code.

## 3. Technology decision

### Desktop shell

**Tauri 2**.

Reason: Rust application core + WebView renderer, small native shell, explicit IPC boundary, and Linux support. Tauri's architecture separates the Rust core from the WebView and provides commands/events/channels for controlled communication.

### UI

**React + TypeScript**.

Reason: strong component model, typed UI contracts, mature rendering ecosystem, animation/layout capability, and suitability for dynamic surface composition.

### Native/core layer

**Rust**.

The core owns privileged/system operations and integrates with existing AIONS/WPC Rust components. The frontend must not directly own filesystem, process, Git, model, or security-sensitive authority.

### Editor

**Monaco Editor**, embedded as a surface when code editing/inspection is required.

### Visualisation

Use SVG/DOM for normal interactive diagrams; Canvas/WebGL where mark count or rendering performance justifies it. The model should emit structured visualization intent/data rather than arbitrary executable rendering code wherever possible.

## 4. Architectural planes

AIONS Studio is split into three logical planes:

### Cognitive plane

- AIONS agent/router
- local Qwen models
- larger/local or remote models
- planning/reasoning
- memory/CBMS
- escalation to stronger models

### Presentation plane

- Presentation Protocol
- Workspace Engine
- Surface lifecycle
- layout/composition
- renderer
- animation/motion
- voice presentation state

### Execution plane

- filesystem
- terminal/PTY
- Git/GitHub
- MCP tools
- CI
- WPC
- CUDA/GPU services
- email/calendar/browser integrations
- system processes

The planes communicate through explicit typed contracts.

## 5. Presentation Protocol

The Presentation Protocol is the primary contract between AIONS and the Studio renderer.

The agent must be able to request operations such as:

- `CREATE_SURFACE`
- `UPDATE_SURFACE`
- `MOVE_SURFACE`
- `RESIZE_SURFACE`
- `FOCUS_SURFACE`
- `GROUP_SURFACES`
- `COLLAPSE_SURFACE`
- `CLOSE_SURFACE`
- `CLEAR_WORKSPACE`

A surface request contains a stable ID, surface type, data payload, semantic priority, preferred presentation hints, and permitted user actions.

The protocol must not expose implementation-specific React components or Rust internals.

## 6. Dynamic Surface model

A Surface is a temporary visual object. It is **not a permanent application slot**.

Initial surface families:

- code/editor
- diff
- terminal
- CI/logs
- dependency/architecture graph
- chart/telemetry
- email
- browser/web content
- video/media
- image
- document
- agent/explanation
- model/runtime status
- memory/KV visualization

Every surface supports lifecycle state, focus state, size/position hints, relationships to other surfaces, and optional actions.

The Workspace Engine decides the final layout. The agent supplies semantic intent; it does not dictate raw pixel coordinates in normal operation.

## 7. Workspace behaviour

The workspace starts intentionally sparse. There is no permanent sidebar, permanent email pane, permanent chat pane, or permanent browser pane.

When a task begins, surfaces materialize with motion and settle into a task-specific composition. When the task changes, the composition can morph rather than forcing the user through application/window switching.

Examples:

- "Show me WPC architecture" -> large interactive architecture graph.
- "Open the decoder source" -> graph remains as context while code surface becomes primary.
- "Show the failing CI" -> CI/log surface takes focus and related source is attached as context.
- "Show the email and the draft reply" -> email and draft surfaces are composed for review.
- "Play this video and show the mixer" -> media surface becomes primary and an audio-control surface appears as context.

## 8. Visual design system

### Identity

AIONS uses a dark, high-end engineering/AI-laboratory aesthetic.

Primary semantic palette:

- near-black graphite: workspace foundation
- deep emerald: AIONS identity / stable system state
- electric green: active/healthy/selected state
- warm amber/yellow: attention, activity, important state, pending decision
- warm off-white: primary text
- neutral graphite/slate: secondary structure

Color must communicate state, not merely decoration.

### Surface language

Surfaces should feel spatial and layered rather than like traditional OS windows:

- restrained translucency
- depth and elevation
- subtle emerald/amber edge lighting
- soft shadows
- controlled blur
- high contrast typography
- rounded but not toy-like geometry
- strong focus treatment

Avoid excessive neon, gaming-RGB styling, gratuitous gradients, and visual noise.

### Motion

Motion is semantic:

- materialize
- expand
- focus
- transform
- collapse
- dismiss

Transitions should feel continuous so that the workspace appears to reorganize itself rather than open/close conventional windows.

Respect reduced-motion preferences.

## 9. Voice-first interaction

Voice is a first-class input path, not a microphone attached to a chat box.

Pipeline:

`voice -> speech-to-text -> AIONS -> action/reasoning -> Presentation Protocol -> workspace`

Responses may be spoken while the workspace visually materializes supporting evidence.

Text interaction remains available for precision, especially terminal commands and code.

## 10. AI-generated visualisation

The local Qwen model may generate structured visualisation specifications, diagrams, SVG, or Canvas/WebGL instructions where appropriate.

Prefer a safe declarative tool interface such as:

- `create_graph`
- `add_node`
- `add_edge`
- `highlight_node`
- `create_chart`
- `render_svg`
- `render_canvas_scene`

The model should not receive unrestricted browser/runtime execution privileges merely to draw a diagram. Rendering and execution authority stay behind AIONS tool boundaries.

For photorealistic/image-model generation, AIONS may later route to a dedicated image model; Qwen 4B is the orchestrator, not the image generator itself.

## 11. Security model

The WebView is untrusted relative to the Rust core. Privileged operations must cross explicit, capability-controlled IPC boundaries.

Never expose unrestricted filesystem/process/GitHub/model credentials to the frontend.

Presentation data must be treated as data, not executable authority.

Approval policies must exist for high-impact actions such as sending email, destructive filesystem operations, pushing/merging code, or external side effects.

## 12. Repository boundaries

`feature/gui` is intentionally branched from `main` and remains independent of Cloud/KV experimental branches.

The existing `arch/studio` and `stage2/studio-contracts-v2` work may be consumed later through explicit contracts. GUI implementation must not mutate or depend on those branches' working state during initial scaffolding.

## 13. Implementation milestones

### M0 — Design baseline

- this specification
- workspace model
- surface model
- Presentation Protocol draft
- visual tokens
- Linux-first policy

### M1 — Minimal Studio shell

- Tauri 2 shell
- React/TypeScript frontend
- Rust core
- one clean workspace
- AIONS identity/presence
- dynamic surface host
- green/amber design tokens

### M2 — Surface engine

- create/update/move/resize/focus/collapse/close
- animated materialisation
- layout engine
- surface relationships
- workspace state persistence

### M3 — First useful surfaces

- Monaco code surface
- terminal surface
- architecture/dependency graph
- diff/log surface

### M4 — AIONS integration

- typed Presentation Protocol transport
- agent-driven workspace actions
- MCP-backed execution
- approval boundaries

### M5 — Browser/media

- browser surface
- video surface
- audio control surface
- media-aware composition

### M6 — Voice

- STT
- TTS
- voice-first commands
- spoken response + visual evidence composition

### M7 — Memory/model integration

- Qwen 4B local routing
- larger-model escalation
- CBMS/memory surfaces
- runtime/KV/WPC visualisation

## 14. First vertical slice acceptance criteria

The first implementation is successful only when all of the following are true:

1. AIONS Studio launches as one Linux desktop application.
2. The workspace starts visually sparse.
3. A surface can be created without a permanent dock slot.
4. At least three surface types can coexist.
5. Surfaces can dynamically change size, position, focus, and lifecycle.
6. The same surface host can render different surface types.
7. The Presentation Protocol drives workspace changes through typed data.
8. The visual identity is dark graphite + emerald + amber/yellow.
9. No Windows-specific implementation is introduced.
10. No existing Cloud/KV branch is modified.
11. Tests cover the workspace state transitions and protocol validation.

## 15. Non-goals for the first slice

- full IDE parity with JetBrains
- full Cursor parity
- full browser engine implementation
- unrestricted autonomous computer control
- photorealistic image generation
- production voice stack
- multi-monitor orchestration
- mobile UI

These are later capabilities, not reasons to complicate the initial architecture.

## 16. Architectural principle

> **The interface is not the application. The workspace is the visual body of AIONS.**

The agent decides what information and tools are relevant. The Presentation Protocol expresses that intent. The Workspace Engine composes the space. The renderer makes it beautiful and interactive. Rust remains responsible for privileged execution and integration with the existing AIONS/WPC system.
