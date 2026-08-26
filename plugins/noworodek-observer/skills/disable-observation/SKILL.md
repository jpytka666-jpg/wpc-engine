---
name: disable-observation
description: Disable the Noworodek Claude Code observer for the current project.
disable-model-invocation: true
---
# Disable Noworodek Observation

Remove or update `${CLAUDE_PROJECT_DIR}/.noworodek/observer.json` so that `enabled` is `false`.

Confirm to the user that no new trace events will be recorded after the change. Existing local traces are not deleted by this operation.
