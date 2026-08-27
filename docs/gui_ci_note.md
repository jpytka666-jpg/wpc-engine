# AIONS Studio CI

The `feature/gui` branch runs `studio-ci.yml` on push and on pull requests touching `studio-core`, `studio-ui`, or the workflow itself.

The frontend gate includes both Vitest and the existing typecheck/build command.
