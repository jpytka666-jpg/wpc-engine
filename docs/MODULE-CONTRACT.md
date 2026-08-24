# AIONS Studio module contract

## Role
System-facing developer environment: editor, compiler integration, debugger, AI agent surface, Git controls, and system inspection.

## UX principle
The IDE is the face of AIONS OS, not a terminal wrapper. System state, memory, agents, and builds are first-class views.

## Boundaries
Use stable APIs to WPC, memory, graph, kernel, and Ghost Gate. Do not own those subsystems.

## Rule
GitHub-first. Local implementation comes only after design/code is accepted and CI is green.
