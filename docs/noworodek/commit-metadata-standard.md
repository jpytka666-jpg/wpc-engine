# Noworodek Commit Metadata Standard

Purpose: prevent `Noworodek` work from being confused with WPC/AIONS work when reviewing the Git history.

## Required commit subject

Every commit made for this project on branch `Noworodek` MUST begin with:

```text
[NOWORODEK][branch=Noworodek][project=wpc-engine]
```

Then add a conventional action, for example:

```text
[NOWORODEK][branch=Noworodek][project=wpc-engine] feat: add low-rank weight representation
```

## Required commit body metadata

The body SHOULD contain:

```text
Author: M. Szul via GPT-5.6 Luna
Project: wpc-engine / Noworodek
Branch: Noworodek
Workstream: <teacher-learning|transformer|weights|wpc-backend|gpu|tokenizer|observatory|docs>
Purpose: <why this change exists>
Outcome: <what changed / what was measured>
Parent: <parent commit SHA>
CI: <pending|green|failed|not-run>
Evidence: <test/log/benchmark path when applicable>
```

The exact GitHub commit SHA is the source-of-truth identifier and is recorded after commit creation when reporting status. Do not fabricate metadata that GitHub has not supplied.

## Separation rule

`Noworodek` commits must not be described as generic WPC or AIONS commits. The branch and project prefix must remain visible in the commit subject so history, diffs, and future automation can distinguish this research line immediately.

## Naming examples

```text
[NOWORODEK][branch=Noworodek][project=wpc-engine] feat: externalize transformer parameters
[NOWORODEK][branch=Noworodek][project=wpc-engine] feat: observe Claude Code tool trajectory
[NOWORODEK][branch=Noworodek][project=wpc-engine] bench: compare WPC fused GEMV and low-rank
[NOWORODEK][branch=Noworodek][project=wpc-engine] docs: record measured non-materializing decode result
```

## Status discipline

A commit is not labelled GREEN merely because it was created successfully. CI/test status must be checked separately. Benchmark claims require a path to the captured evidence.
