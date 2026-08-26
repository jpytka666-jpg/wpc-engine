#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const hookName = process.argv[2] ?? 'Unknown';
const input = fs.readFileSync(0, 'utf8').trim();
if (!input) process.exit(0);

const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const configPath = path.join(projectDir, '.noworodek', 'observer.json');

let config;
try {
  config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
} catch {
  // Observer is deliberately opt-in. Installation alone must not collect data.
  process.exit(0);
}

if (config?.enabled !== true) process.exit(0);

let payload;
try {
  payload = JSON.parse(input);
} catch {
  payload = { raw_stdin: input.slice(0, 262144) };
}

const sensitiveKey = /(api[_-]?key|access[_-]?token|auth(orization)?|password|passwd|secret|private[_-]?key|cookie|credential)/i;
const secretValue = /(sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{12,})/g;

function redact(value, depth = 0) {
  if (depth > 12) return '[MAX_DEPTH]';
  if (typeof value === 'string') {
    return value.replace(secretValue, '[REDACTED_SECRET]').slice(0, 262144);
  }
  if (Array.isArray(value)) return value.map((item) => redact(item, depth + 1));
  if (value && typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] = sensitiveKey.test(key) ? '[REDACTED]' : redact(item, depth + 1);
    }
    return out;
  }
  return value;
}

const event = redact(payload);
const sessionId = String(event.session_id ?? event.sessionId ?? 'unknown-session');
const safeSession = sessionId.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 160) || 'unknown-session';
const day = new Date().toISOString().slice(0, 10);

const dataRoot = process.env.CLAUDE_PLUGIN_DATA || path.join(projectDir, '.noworodek', 'observer-data');
const traceDir = path.join(dataRoot, 'traces', day);
fs.mkdirSync(traceDir, { recursive: true });

const record = {
  schema_version: 1,
  observed_at: new Date().toISOString(),
  hook: hookName,
  provider: 'claude-code',
  project_dir: projectDir,
  session_id: sessionId,
  payload: event
};

fs.appendFileSync(
  path.join(traceDir, `${safeSession}.jsonl`),
  `${JSON.stringify(record)}\n`,
  { encoding: 'utf8' }
);
