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

  // A turn with nothing but a question is not a lesson - nothing was learned from it.
  if (lines.length > 1) {
    turns += 1;
    out.push(lines.join('\n'));
  }
}

fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
const text = out.join('\n\n');
fs.writeFileSync(outPath, text, 'utf8');

console.log(`sladow przeczytanych : ${jsonlFiles(traceRoot).length} plikow`);
console.log(`epizodow             : ${lessons.size}`);
console.log(`lekcji zapisanych    : ${turns}   (epizody bez zadnego dzialania pominiete)`);
console.log(`dzialan              : ${actions}`);
console.log(`w tym z werdyktem    : ${withVerdict}   <- to sa najcenniejsze`);
console.log(`znakow               : ${text.length}`);
console.log(`zapisane             : ${outPath}`);
console.log();
console.log('dalej: cbms <ksiazka> build <ten plik> <nowa-ksiazka>   aby dopisac slownictwo pracy');
console.log('       cbms <nowa-ksiazka> ids <ten plik> <plik.u16>    aby zamienic na numery');
