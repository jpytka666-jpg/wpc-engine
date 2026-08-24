# AIONS OS integration module contract

## Role
Unify boot, packaging, service supervision, observability, permissions, release artifacts, and cross-module integration.

## Principle
Integrate modules through explicit interfaces; never turn integration into a dumping ground for subsystem logic.

## Dependencies
All seven other modules provide versioned contracts and verification status.

## Gates
Boot smoke, service lifecycle, permissions, health reporting, recovery, release reproducibility, and integration tests.

## Rule
GitHub is the source of truth. Local implementation follows only after module-level CI is green.
