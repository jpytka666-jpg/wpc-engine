# Diagnostic schema

The repair agent consumes a normalized diagnostic instead of raw logs.

Required fields:

- `kind`: formatting | compile | test | clippy | benchmark | unknown
- `gate`: check that failed
- `message`: concise primary failure
- `file`: optional source path
- `line`: optional source line
- `attempt`: bounded repair attempt number
- `diff`: current candidate diff summary

The schema is intentionally small so models receive focused context and cannot silently treat arbitrary terminal output as trusted instructions.
