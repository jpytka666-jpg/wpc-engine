# Memory / System Graph module contract

## Role
Interactive graph of AI memory, code, processes, agents, dependencies, and system relationships.

## Design
Graph is an operational interface, not decoration. Nodes and edges must map to real entities and stable IDs.

## Boundaries
Consume events and metadata from other modules; do not become the owner of their state.

## Rule
GitHub-first design and implementation. Local visualization follows only after interfaces and CI are accepted.
