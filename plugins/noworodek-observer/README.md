# Noworodek Teacher Observer

Claude Code plugin for the Noworodek learning pipeline.

## What it captures

The plugin records observable Claude Code lifecycle events, including user prompt submission, tool calls/results, subagent lifecycle, compaction, stop, and session boundaries. It stores an append-only JSONL raw trace locally so the trace can later be normalized into training experiences.

It does **not** capture private chain-of-thought, does not mutate the teacher, and does not upload data remotely.

## Explicit opt-in

Installation does not enable collection. Observation remains disabled until the project contains:

```json
{
  "enabled": true,
  "training": false,
  "storage": "local-jsonl"
}
```

Use `/noworodek-observer:enable-observation` in Claude Code after explicitly deciding to record the session. Use `/noworodek-observer:disable-observation` to stop future capture. Existing traces are retained unless the user deletes them.

## Storage

The plugin writes traces beneath `${CLAUDE_PLUGIN_DATA}/traces/<date>/`. This keeps runtime data outside the plugin installation/cache, which is ephemeral by design.

## Installation from this repository

After cloning the repository:

```text
/plugin marketplace add ./path/to/wpc-engine
/plugin install noworodek-observer@noworodek --scope local
```

Or from the project directory, add the repository as a Git marketplace and install the plugin at project scope.

## Security model

The observer is intentionally local-only in this phase. Hook scripts run with the same user permissions as Claude Code, so review the plugin source before enabling it. The recorder redacts common API-key, token, authorization, password, secret, private-key, cookie, and credential fields and caps individual strings to 256 KiB.

## Training boundary

The plugin is the **teacher-observation ingress**, not the trainer itself. The Noworodek trainer consumes normalized experiences after evaluator approval. This separation lets us replay and re-evaluate the same raw trace without changing what was originally observed.
