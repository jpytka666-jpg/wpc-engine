# Noworodek Git History Rewrite — 2026-08-26

## Purpose

The `Noworodek` branch contained a long sequence of implementation commits with inconsistent commit-message formats. The branch history is being condensed into canonical milestone commits so that Noworodek work is unambiguous and cannot be confused with unrelated WPC/AIONS work.

## Safety

- `main` is not modified.
- Pre-rewrite work remains reachable from `archive/Noworodek-pre-metadata-rewrite-2026-08-26`.
- Original pre-rewrite HEAD: `5c2bb3751858c43f6c161c8611458317b4567190`.
- Selected milestone trees are preserved exactly in the canonical chain.

## Canonical milestones

- FOUNDATION: `27ffc2559992b638b508fbf5df65093acf0cc8ed`
- TEACHER: `b910e9292dd3b190d52eec3fe60bcfa55a7dc903`
- GPU: `9891585fec7f7a6f09a05ac55ad52cc42bb9d952`
- TOKENIZER: `b680b46eab17310013f6408b60ca32a4e9a312f4`
- TRANSFORMER: `a18b6527a11b33cc86191adf085e6ffcb1b1deef`
- TRAINING: `d7e5285d527b13934473c4946c6d6c8a746bf95e`
- REPRESENTATION: `4f039c0ea253da79a9fccb4757281d4e3339e797`
- LOWRANK: `965786787ebc05242d0c12e6d7663b5dacd8d792`
- META: `515dba5788e1888c9260d40b5790770abdc0040f`
- MATH: `5c2bb3751858c43f6c161c8611458317b4567190`

The canonical branch intentionally uses milestone commits; the archive retains the original granular history.
