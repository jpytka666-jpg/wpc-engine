# Claude handoff — AIONS OS integration

This branch owns cross-module integration, packaging, boot, service supervision, permissions and release gates.
It must not absorb subsystem logic; consume stable contracts from the seven other branches.
Do not modify unrelated modules or existing local AIONS workspaces.
GitHub first; local integration only after module-level CI is green.
