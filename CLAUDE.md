# Claude handoff — Memory / KV

This branch owns hot KV, compressed KV research, and CBMS memory interfaces.
Keep generation-critical data separate from persistent storage paths.
Do not modify unrelated modules or existing local AIONS workspaces.
GitHub first; local implementation only after CI/benchmark acceptance.
