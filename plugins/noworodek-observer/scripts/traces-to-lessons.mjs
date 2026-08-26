#!/usr/bin/env node
/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-26 21:16:45
 * REASON FOR CREATION: Noworodek may only ever be fed CBMS. The observer writes JSON, and
 *   JSON braces are not a lesson - a learner given raw trace structure would spend its
 *   capacity on punctuation. This renders each turn as the thing worth learning: what was
 *   asked, what was attempted, and what came back.
 * MECHANICS: Groups trace lines by episode, orders them, and writes one plain-text lesson
 *   per turn. Only the fields that carry a lesson survive: the tool, the essence of its
 *   input, the outcome, and the runner's verdict where there was one. Everything else -
 *   ids, timestamps, JSON scaffolding - is dropped, because it is bookkeeping and would
 *   only teach the model to reproduce bookkeeping. The output then goes through the code
 *   book like any other text: `cbms build` to extend the vocabulary, `cbms ids` to encode.
 * SYSTEM PART: Noworodek, teacher-observation lane.
 * ARCHITECTURE FUNCTION: The step that puts a working session on the same path as every
 *   other corpus - text, then codebook, then ids. Nothing reaches the model except CBMS.
 * DEPENDENCIES/LINKS: reads the JSONL written by record-event.mjs; output consumed by
 *   cbms-writing's `build` and `ids` commands.
 * TECH STACK: Node, standard library. Node because the traces are JSON and were written
 *   by Node; parsing them elsewhere would mean adding a JSON dependency to a crate that
 *   deliberately has none.
 * LOCAL WORKSPACE: plugins/noworodek-observer/scripts/traces-to-lessons.mjs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
 * ==========================================
 */

import fs from 'node:fs';
import path from 'node:path';

const [, , traceRoot, outPath] = process.argv;
if (!traceRoot || !outPath) {
  console.error('traces-to-lessons.mjs <katalog-sladow> <plik-wyjsciowy.txt>');
  process.exit(2);
}

function jsonlFiles(root) {
  const out = [];
  const walk = (dir) => {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith('.jsonl')) out.push(p);
    }
  };
  walk(root);
  return out.sort();
}

/** One line of a command, without the ceremony. A full shell line is mostly paths and
 *  flags that will never repeat; the verb and its object are what generalise. */
function essence(record) {
  const input = record.payload?.tool_input ?? {};
  if (typeof input.command === 'string') {
    return input.command.replace(/\s+/g, ' ').trim().slice(0, 200);
  }
  if (typeof input.file_path === 'string') {
    // The name matters, the machine-specific path does not.
    return path.basename(input.file_path);
  }
  if (typeof input.pattern === 'string') return input.pattern.slice(0, 120);
  if (typeof input.query === 'string') return input.query.slice(0, 120);
  if (typeof input.prompt === 'string') return input.prompt.replace(/\s+/g, ' ').slice(0, 400);
  return '';
}

/* ---------- the quality gate ---------------------------------------------
 *
 * AIONS already made this mistake once, one level down. Its memory decided what to keep
 * by looking at the SHAPE OF THE QUESTION rather than at whether anything had been
 * learned, and filled 73% of itself with its own echo - 455 blocks quarantined. Feeding
 * a model unfiltered traces repeats that in the weights, where it cannot be quarantined
 * afterwards.
 *
 * So the same defence, one level up: named rules, and every rejection written down.
 * A gate whose refusals are invisible is a gate nobody can argue with.
 *
 * Kill switch: NOWORODEK_LESSON_GATE=0
 */

const GATE_OFF = process.env.NOWORODEK_LESSON_GATE === '0';

// Tools that only look around. Looking is not learning: nothing changed, nothing was
// verified, and a model trained on it learns to look around.
const PASSIVE = new Set(['Read', 'Glob', 'Grep', 'LS', 'NotebookRead', 'TodoWrite']);

// Files that ARE the observer and the learner. A lesson about building the thing that is
// watching is the echo problem wearing a different hat: it would train on itself.
const SELF = /(record-event|traces-to-lessons|train-cbms|kapral|noworodek-observer|scale-probe)/i;

const seenLessons = new Set();
const rejections = [];

function gate(records, text) {
  if (GATE_OFF) return { ok: true, reason: 'gate_off' };

  const done = records.filter((r) => r.hook?.startsWith('PostToolUse'));
  if (done.length === 0) return { ok: false, reason: 'R1_zadnego_dzialania' };

  if (text.length < 40) return { ok: false, reason: 'R2_za_krotkie' };

  if (done.every((r) => PASSIVE.has(r.tool))) {
    return { ok: false, reason: 'R3_samo_rozgladanie' };
  }

  const selfHits = done.filter((r) => SELF.test(JSON.stringify(r.payload?.tool_input ?? ''))).length;
  if (selfHits > done.length / 2) {
    return { ok: false, reason: 'R4_obserwuje_sam_siebie' };
  }

  // Something objective must have happened: a runner gave a verdict, or a failure was
  // followed by a success. Without either, the episode records activity, not learning.
  const hasVerdict = done.some((r) => r.verification);
  const outcomes = done.map((r) => r.outcome);
  const repaired = outcomes.indexOf('error') !== -1
    && outcomes.lastIndexOf('ok') > outcomes.indexOf('error');
  if (!hasVerdict && !repaired) {
    return { ok: false, reason: 'R5_nic_sprawdzalnego' };
  }

  const fingerprint = text.replace(/\d+/g, '#');
  if (seenLessons.has(fingerprint)) return { ok: false, reason: 'R6_powtorka' };
  seenLessons.add(fingerprint);

  return { ok: true, reason: 'ok' };
}

const lessons = new Map(); // episode -> lines

for (const file of jsonlFiles(traceRoot)) {
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let r;
    try {
      r = JSON.parse(line);
    } catch {
      continue;
    }
    const key = r.episode ?? `${r.session_id}#0`;
    if (!lessons.has(key)) lessons.set(key, []);
    lessons.get(key).push(r);
  }
}

const out = [];
let turns = 0;
let actions = 0;
let withVerdict = 0;

for (const [, records] of lessons) {
  records.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
  const lines = [];

  for (const r of records) {
    if (r.hook === 'UserPromptSubmit') {
      const asked = essence(r) || r.payload?.prompt || '';
      if (asked) lines.push(`ZADANIE ${asked}`);
      continue;
    }
    // Only completed actions are lessons. A start with no end teaches nothing, and an
    // action recorded twice would teach that everything happens twice.
    if (!r.hook?.startsWith('PostToolUse')) continue;

    const what = essence(r);
    const bits = [`ROBI ${r.tool ?? 'nieznane'}`];
    if (what) bits.push(what);
    if (r.outcome) bits.push(`WYNIK ${r.outcome === 'ok' ? 'dobrze' : 'blad'}`);
    if (r.verification) {
      const v = r.verification;
      bits.push(
        `SPRAWDZENIE ${v.kind} ${v.ok ? 'przeszlo' : 'padlo'}` +
          (v.passed !== undefined ? ` ${v.passed} zdanych ${v.failed} padlych` : '')
      );
      withVerdict += 1;
    }
    lines.push(bits.join(' '));
    actions += 1;
  }

  const text = lines.join('\n');
  const verdict = gate(records, text);
  if (!verdict.ok) {
    rejections.push({
      at: new Date().toISOString(),
      reason: verdict.reason,
      actions: records.filter((r) => r.hook?.startsWith('PostToolUse')).length,
      preview: text.slice(0, 200),
    });
    continue;
  }
  turns += 1;
  out.push(text);
}

fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
const text = out.join('\n\n');
fs.writeFileSync(outPath, text, 'utf8');

console.log(`sladow przeczytanych : ${jsonlFiles(traceRoot).length} plikow`);
console.log(`epizodow             : ${lessons.size}`);
console.log(`lekcji PRZYJETYCH    : ${turns}`);
console.log(`lekcji ODRZUCONYCH   : ${rejections.length}${GATE_OFF ? '   (BRAMKA WYLACZONA)' : ''}`);
if (rejections.length) {
  const byReason = {};
  for (const r of rejections) byReason[r.reason] = (byReason[r.reason] ?? 0) + 1;
  for (const [reason, n] of Object.entries(byReason).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${String(n).padStart(4)}  ${reason}`);
  }
  // Written down, because a gate whose refusals are invisible is a gate nobody can argue
  // with - and the one time it refuses something valuable, this is how it gets found.
  const logPath = path.join(path.dirname(path.resolve(outPath)), 'odrzucone-lekcje.jsonl');
  fs.appendFileSync(logPath, rejections.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');
  console.log(`    dziennik odrzucen: ${logPath}`);
}
console.log(`dzialan              : ${actions}`);
console.log(`w tym z werdyktem    : ${withVerdict}   <- to sa najcenniejsze`);
console.log(`znakow               : ${text.length}`);
console.log(`zapisane             : ${outPath}`);
console.log();
console.log('dalej: cbms <ksiazka> build <ten plik> <nowa-ksiazka>   aby dopisac slownictwo pracy');
console.log('       cbms <nowa-ksiazka> ids <ten plik> <plik.u16>    aby zamienic na numery');
