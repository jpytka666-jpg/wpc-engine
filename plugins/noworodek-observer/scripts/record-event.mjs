#!/usr/bin/env node
/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 20:54:51
 * REASON FOR CREATION: The first version recorded events. Events are not lessons. A tool
 *   call and its result arrived as two unrelated lines, so a learner could not tell what
 *   any action produced, how long it took, or whether it worked. This version records the
 *   thing worth learning from: an action, its outcome, and - where the session shows one -
 *   the correction that followed a failure.
 * MECHANICS: Pairs PreToolUse with PostToolUse through the tool-use id, so each action
 *   carries its own duration and outcome. Groups events into turns, so a learner can see
 *   an attempt and its repair as one episode rather than as scattered lines. Extracts
 *   pass/fail counts from build and test output, which is the only label in a coding
 *   session that is not a matter of opinion. Caps payloads and records a digest of what
 *   was cut, so a long session stays trainable instead of becoming a disk of transcripts.
 * SYSTEM PART: Noworodek, teacher-observation lane.
 * ARCHITECTURE FUNCTION: Turns a working session into supervised experience. Everything
 *   downstream - normalisation, curriculum, training - reads this format.
 * DEPENDENCIES/LINKS: Claude Code hook protocol; consent file .noworodek/observer.json;
 *   writes JSONL under CLAUDE_PLUGIN_DATA or .noworodek/observer-data.
 * TECH STACK: Node, standard library only. Node because the hook contract and the plugin
 *   were already written for it, and a second runtime here would buy nothing.
 * LOCAL WORKSPACE: plugins/noworodek-observer/scripts/record-event.mjs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
 * ==========================================
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

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
  payload = { raw_stdin: input.slice(0, 8192) };
}

/* ---------- redaction ---------------------------------------------------- */

const sensitiveKey = /(api[_-]?key|access[_-]?token|auth(orization)?|password|passwd|secret|private[_-]?key|cookie|credential)/i;
const secretValue = /(sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)/g;
// A file whose whole purpose is to hold secrets should never have its contents recorded,
// however innocent any individual line looks.
const secretFile = /(^|[\\/])(\.env(\.|$)|id_rsa|id_ed25519|\.pem$|\.pfx$|credentials(\.|$)|\.netrc$)/i;

// Payloads are capped hard. The previous limit of 256 KB per string meant a session could
// write hundreds of megabytes, and a trace too big to load is a trace nobody trains on.
const MAX_STRING = 8192;

function digestOf(text) {
  return crypto.createHash('sha256').update(text).digest('hex').slice(0, 16);
}

function redact(value, depth = 0) {
  if (depth > 12) return '[MAX_DEPTH]';
  if (typeof value === 'string') {
    const cleaned = value.replace(secretValue, '[REDACTED_SECRET]');
    if (cleaned.length <= MAX_STRING) return cleaned;
    // Keep both ends: the start says what it is, the end usually says how it went.
    const head = cleaned.slice(0, MAX_STRING * 0.75);
    const tail = cleaned.slice(-MAX_STRING * 0.25);
    return `${head}\n...[CUT ${cleaned.length - MAX_STRING} chars, sha ${digestOf(cleaned)}]...\n${tail}`;
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

/* ---------- outcome, the label a learner needs --------------------------- */

/** Build and test runners state their own verdict. That verdict is the one signal in a
 *  coding session that is not a matter of opinion, so it is lifted out and stored plainly
 *  instead of being left buried in output nobody parses twice. */
function verificationOf(text) {
  if (typeof text !== 'string' || !text) return null;
  const out = {};
  let found = false;

  const cargo = text.match(/test result:\s*(ok|FAILED)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed/);
  if (cargo) {
    out.kind = 'cargo-test';
    out.passed = Number(cargo[2]);
    out.failed = Number(cargo[3]);
    out.ok = cargo[1] === 'ok';
    found = true;
  }
  const pytest = text.match(/(\d+)\s+passed(?:,\s*(\d+)\s+failed)?/);
  if (!found && pytest) {
    out.kind = 'pytest';
    out.passed = Number(pytest[1]);
    out.failed = Number(pytest[2] ?? 0);
    out.ok = out.failed === 0;
    found = true;
  }
  const rustc = text.match(/^error(\[[A-Z0-9]+\])?:/m);
  if (!found && rustc) {
    out.kind = 'compile';
    out.ok = false;
    out.first_error = text.match(/^error(\[[A-Z0-9]+\])?:.*$/m)?.[0]?.slice(0, 300) ?? null;
    found = true;
  }
  if (!found && /\b(Compiling|Finished)\b.*\b(dev|release)\b/.test(text) && !/^error/m.test(text)) {
    out.kind = 'compile';
    out.ok = true;
    found = true;
  }
  return found ? out : null;
}

/* ---------- episode state ------------------------------------------------ */

const event = redact(payload);
const sessionId = String(event.session_id ?? event.sessionId ?? 'unknown-session');
const safeSession = sessionId.replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 160) || 'unknown-session';
const day = new Date().toISOString().slice(0, 10);

const dataRoot = process.env.CLAUDE_PLUGIN_DATA || path.join(projectDir, '.noworodek', 'observer-data');
const traceDir = path.join(dataRoot, 'traces', day);
fs.mkdirSync(traceDir, { recursive: true });

// Small per-session state: a sequence counter, the current turn, and the start time of
// each in-flight tool call. Without it every event is an orphan.
const statePath = path.join(dataRoot, `state-${safeSession}.json`);
let state = { seq: 0, turn: 0, pending: {} };
try {
  state = { ...state, ...JSON.parse(fs.readFileSync(statePath, 'utf8')) };
} catch { /* first event of the session */ }

// A turn begins when the user speaks. Everything until the next prompt belongs to it,
// which is what makes an attempt and its repair one episode rather than two lines.
if (hookName === 'UserPromptSubmit') state.turn += 1;
state.seq += 1;

const toolUseId =
  event.tool_use_id ?? event.toolUseId ?? event.tool_use?.id ?? null;
const toolName = event.tool_name ?? event.toolName ?? null;

let durationMs = null;
if (toolUseId) {
  if (hookName === 'PreToolUse') {
    state.pending[toolUseId] = Date.now();
  } else if (hookName.startsWith('PostToolUse')) {
    const started = state.pending[toolUseId];
    if (started) {
      durationMs = Date.now() - started;
      delete state.pending[toolUseId];
    }
  }
}

// Bounded: a session that never posts a result must not grow this without limit.
const pendingKeys = Object.keys(state.pending);
if (pendingKeys.length > 200) {
  for (const k of pendingKeys.slice(0, pendingKeys.length - 200)) delete state.pending[k];
}

/* ---------- the record --------------------------------------------------- */

const responseText = [
  event.tool_response, event.toolResponse, event.tool_result, event.output, event.stdout,
].find((v) => typeof v === 'string');
const responseBlob = responseText ?? (event.tool_response ? JSON.stringify(event.tool_response) : '');

const verification = verificationOf(responseBlob);

// A runner's own verdict beats any guess made from its output. The first version of this
// looked for the word "error" and called a cargo run "ok" while the same line said
// FAILED - a label that lies is worse than no label, because a learner believes it.
let outcome = null;
if (hookName === 'PostToolUseFailure') outcome = 'error';
else if (hookName === 'PostToolUse') {
  if (verification) outcome = verification.ok ? 'ok' : 'error';
  else outcome = /(^|\n)\s*(error|Error|ERROR|FAILED|failed)[: ]/.test(responseBlob) ? 'error' : 'ok';
}

// A file that exists to hold secrets is noted, never quoted.
const targetPath = event.tool_input?.file_path ?? event.tool_input?.path ?? null;
const redactedFile = targetPath && secretFile.test(String(targetPath));

const record = {
  schema_version: 2,
  observed_at: new Date().toISOString(),
  hook: hookName,
  provider: 'claude-code',
  project_dir: projectDir,
  session_id: sessionId,

  // What makes a trace learnable rather than merely stored.
  seq: state.seq,
  turn: state.turn,
  episode: `${safeSession}#${state.turn}`,
  tool_use_id: toolUseId,
  tool: toolName,
  duration_ms: durationMs,
  outcome,
  verification,
  agent: event.agent_id ?? event.agentId ?? null,

  payload: redactedFile
    ? { ...event, tool_response: '[REDACTED_SECRET_FILE]', tool_input: { ...event.tool_input, content: '[REDACTED_SECRET_FILE]' } }
    : event,
};

fs.appendFileSync(
  path.join(traceDir, `${safeSession}.jsonl`),
  `${JSON.stringify(record)}\n`,
  { encoding: 'utf8' }
);

try {
  fs.writeFileSync(statePath, JSON.stringify(state), { encoding: 'utf8' });
} catch { /* a lost counter costs ordering, not correctness; never fail the session */ }
