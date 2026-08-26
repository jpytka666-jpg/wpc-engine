---
name: enable-observation
description: Enable the Noworodek Claude Code observer for the current project after the user explicitly requests observation.
disable-model-invocation: true
---
# Enable Noworodek Observation

Enable only after the user explicitly asks to start recording the Claude Code session for Noworodek training.

Create `${CLAUDE_PROJECT_DIR}/.noworodek/observer.json` with:

```json
{
  "enabled": true,
  "training": false,
  "storage": "local-jsonl"
}
```

Tell the user that observation is now enabled, where traces are written, and that the current implementation records observable session/tool events only; it does not capture private chain-of-thought and does not remotely upload data.
