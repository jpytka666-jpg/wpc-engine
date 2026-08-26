# Noworodek Commit Metadata Standard

All commits on the `Noworodek` branch must be self-identifying.

## Subject

```text
[WPC-ENGINE][NOWORODEK][WORKSTREAM] type(scope): summary
```

Allowed `type` values: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`, `perf`.

`WORKSTREAM` is uppercase: `FOUNDATION`, `TEACHER`, `GPU`, `TOKENIZER`, `TRANSFORMER`, `TRAINING`, `REPRESENTATION`, `LOWRANK`, `META`, `MATH`, or another explicitly named stream.

## Required body

```text
Project: wpc-engine
Branch: Noworodek
Workstream: WORKSTREAM
Change: <what changed>
Parent: <parent commit SHA>
CI: <pending|green|failed|historical|not-run>
Evidence: <path or none>
```

`CI: pending` is valid when the commit is created and must be replaced/documented after verification.

## Enforcement

`.githooks/commit-msg` rejects commits on `Noworodek` that do not satisfy the subject format or omit the required metadata fields.

## Historical rewrite

On 2026-08-26 the pre-existing Noworodek microcommit chain was condensed into canonical milestone commits. The original pre-rewrite head is preserved by `archive/Noworodek-pre-metadata-rewrite-2026-08-26`. `main` is untouched.
