/**
 * replay.js  -  pure, DOM-free logic for the recorded replay (data/replay.json,
 * schema in playground/public/data/README.md). Unit-tested in test/replay.test.js.
 *
 * The replay has one shared cursor: a token index in [0, replayLength(case)].
 * "cursor = k" means the first k tokens of each run have been written (a run
 * shorter than k is simply finished). Everything the page shows  -  pane text,
 * timeline, repetition meter, counters  -  is a function of (case, cursor).
 */
import { createLoopMeter } from './lib.js';

/** Playback rate at 1x, in tokens per second (roughly a fast local GPU). */
export const TOKENS_PER_SECOND_1X = 60;
/** Offered speed multipliers. */
export const SPEEDS = [1, 4, 16, 32];
/**
 * Autoplay speed: at 32x (1,920 tok/s) the default case's base loop starts
 * after ~2.5 s and the edited run stops after ~3.7 s, so both moments land
 * within the ~5 s first-visit budget. 1x/4x/16x stay one click away.
 */
export const DEFAULT_SPEED = 32;
/** Loop meter settings shared with the live run (same metric in both places). */
export const LOOP_METER = { n: 4, window: 80 };

const MAX_FRAME_MS = 100; // a background tab must not jump the replay ahead

/* ---------- clock ---------- */

export function tokensPerSecond(speed) {
  return TOKENS_PER_SECOND_1X * speed;
}

/** Wall time (ms) to play `cursor` tokens from 0 at `speed`. */
export function cursorToMs(cursor, speed) {
  return (cursor / tokensPerSecond(speed)) * 1000;
}

/** Tokens (fractional) played after `ms` at `speed`. */
export function msToCursor(ms, speed) {
  return (ms / 1000) * tokensPerSecond(speed);
}

/** Advance a fractional cursor by one animation frame; clamped to [0, end]. */
export function advanceCursor(cursor, dtMs, speed, end) {
  const dt = Math.max(0, Math.min(MAX_FRAME_MS, dtMs));
  return clampCursor(cursor + msToCursor(dt, speed), end);
}

export function clampCursor(cursor, end) {
  const c = Number(cursor);
  if (!Number.isFinite(c)) return 0;
  return Math.max(0, Math.min(end, c));
}

/** Token count of the longer run: the replay's end. */
export function replayLength(c) {
  return Math.max(c.base.nTokens, c.edited.nTokens);
}

/** Max-token cap the case was recorded with. */
export function capOf(c) {
  return c.sampling?.max_tokens ?? null;
}

/* ---------- per-run preparation ---------- */

/**
 * Precompute what is needed to answer "state at cursor" cheaply: the joined
 * text, char offsets per token, and the loop-meter score after every token
 * (the live run's createLoopMeter fed token by token, so the meter means the
 * same thing in live and replay).
 */
export function prepareRun(run) {
  const n = run.tokens.length;
  const offsets = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) offsets[i + 1] = offsets[i] + run.tokens[i].length;
  const meter = createLoopMeter(LOOP_METER);
  const repetition = new Float32Array(n);
  for (let i = 0; i < n; i++) repetition[i] = meter.push(run.tokens[i]);
  if (n) repetition[n - 1] = meter.flush(); // the run is over: count the trailing word
  const loops = [...(run.loops ?? [])].sort((a, b) => a.start - b.start);
  return { ...run, loops, text: run.tokens.join(''), offsets, repetition };
}

export function prepareCase(c) {
  return { ...c, base: prepareRun(c.base), edited: prepareRun(c.edited) };
}

/** Text of tokens [from, to) of a prepared run. */
export function textOf(run, from, to) {
  return run.text.slice(run.offsets[from], run.offsets[to]);
}

/** Loop-meter score (0..1) after `cursor` tokens. */
export function repetitionAt(run, cursor) {
  const k = Math.min(Math.floor(cursor), run.repetition.length);
  return k <= 0 ? 0 : run.repetition[k - 1];
}

/* ---------- loops ---------- */

/**
 * The recorded loop segment the run is inside at `cursor` (its last written
 * token lies in [start, end)), or null. A segment is verbatim repetition of
 * the `period` tokens before `start`; `repeats` counts completed copies so
 * far (it ends at floor(segment.repeats)).
 */
export function loopAt(run, cursor) {
  const k = Math.min(Math.floor(cursor), run.nTokens);
  if (k <= 0) return null;
  const last = k - 1;
  const segment = (run.loops ?? []).find((s) => s.start <= last && last < s.end);
  if (!segment) return null;
  const written = k - segment.start;
  return {
    segment,
    written,
    repeats: Math.floor(written / segment.period),
    sourceStart: Math.max(0, segment.start - segment.period),
    currentStart: segment.start + Math.floor((written - 1) / segment.period) * segment.period,
  };
}

/** Tokens written inside recorded loop segments up to `cursor`. */
export function loopTokensAt(run, cursor) {
  const k = Math.min(Math.floor(cursor), run.nTokens);
  let sum = 0;
  for (const s of run.loops ?? []) sum += Math.max(0, Math.min(k, s.end) - s.start);
  return sum;
}

/**
 * A readable snippet of the repeated phrase: the longest line of the source
 * unit, list markers stripped, whitespace collapsed, truncated to `max` chars.
 */
export function loopPhrase(run, segment, max = 64) {
  const unit = textOf(run, Math.max(0, segment.start - segment.period), segment.start);
  const lines = unit
    .split('\n')
    .map((l) => l.replace(/^\s*(?:[*\-•]|\d+\.)\s+/, '').replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  const best = lines.reduce((a, b) => (b.length > a.length ? b : a), '');
  return best.length > max ? best.slice(0, max - 1).trimEnd() + '…' : best;
}

/* ---------- text spans for the panes ---------- */

const CLS = ['', 'loop-src', 'loop-old', 'loop-now'];

/**
 * Split the first `cursor` tokens into styled spans: region "think" before
 * thinkEnd, "answer" after it (the "</think>" token itself is dropped), and
 * class "loop-old" inside recorded loop segments, with  -  for the segment the
 * cursor is in  -  "loop-src" on the original phrase and "loop-now" on the copy
 * being written. Returns [{ region, cls, from, to }] (token indices).
 */
export function textSpans(run, cursor) {
  const k = Math.min(Math.floor(cursor), run.nTokens);
  if (k <= 0) return [];
  const cls = new Uint8Array(k);
  for (const s of run.loops ?? []) {
    for (let i = s.start; i < Math.min(k, s.end); i++) cls[i] = 2;
  }
  const active = loopAt(run, k);
  if (active) {
    for (let i = active.sourceStart; i < active.segment.start; i++) if (cls[i] === 0) cls[i] = 1;
    for (let i = active.currentStart; i < k; i++) cls[i] = 3;
  }
  const te = run.thinkEnd;
  const spans = [];
  let cur = null;
  for (let i = 0; i < k; i++) {
    if (te != null && i === te) {
      cur = null; // the "</think>" token is shown as the think/answer split, not as text
      continue;
    }
    const region = te != null && i > te ? 'answer' : 'think';
    const c = CLS[cls[i]];
    if (cur && cur.region === region && cur.cls === c) cur.to = i + 1;
    else spans.push((cur = { region, cls: c, from: i, to: i + 1 }));
  }
  return spans;
}

/* ---------- counters ---------- */

/** Everything the page shows about one run at `cursor`. */
export function runStateAt(run, cursor) {
  const spent = Math.min(Math.floor(cursor), run.nTokens);
  const done = spent >= run.nTokens;
  const phase = done ? 'done' : run.thinkEnd != null && spent > run.thinkEnd ? 'answer' : 'thinking';
  return {
    spent,
    total: run.nTokens,
    done,
    phase,
    finish: done ? run.finish : null,
    answer: done ? run.answer : null,
    correct: done ? run.correct : null,
    loop: loopAt(run, spent),
    loopTokens: loopTokensAt(run, spent),
  };
}

/** One-line verdict for a finished run, from the data only. */
export function runVerdict(run, cap) {
  const n = run.nTokens.toLocaleString('en-US');
  if (run.finish === 'stop') {
    if (run.answer == null) return `stopped after ${n} tokens · no boxed answer`;
    return `answered ${run.answer} ${run.correct ? '✓' : '✗'} in ${n} tokens`;
  }
  const c = (cap ?? run.nTokens).toLocaleString('en-US');
  return `hit the ${c}-token cap · no answer`;
}

/** Counters strip state at `cursor` (all counts derived from the recording). */
export function countersAt(c, cursor) {
  const base = runStateAt(c.base, cursor);
  const edited = runStateAt(c.edited, cursor);
  const end = replayLength(c);
  const cap = capOf(c);
  return {
    cursor: Math.min(Math.floor(cursor), end),
    end,
    base,
    edited,
    wastedLooping: base.loopTokens + edited.loopTokens,
    verdict:
      base.done && edited.done
        ? { base: runVerdict(c.base, cap), edited: runVerdict(c.edited, cap) }
        : null,
  };
}

/* ---------- case selection + URL state ---------- */

/**
 * How dramatic a loop case is: the base run's longest verbatim loop relative
 * to the edited run's length (a long loop and a short escape score high).
 * Non-loop cases score 0.
 */
export function caseDramaScore(c) {
  if (c.kind !== 'loop' || c.base.finish !== 'length' || c.edited.finish !== 'stop' || !c.edited.correct) return 0;
  const longest = Math.max(0, ...(c.base.loops ?? []).map((s) => s.end - s.start));
  return longest / Math.max(1, c.edited.nTokens);
}

/** Default case: the most dramatic loop case, else the first case. */
export function defaultCaseId(cases) {
  if (!cases.length) return null;
  let best = cases[0];
  let bestScore = caseDramaScore(best);
  for (const c of cases.slice(1)) {
    const s = caseDramaScore(c);
    if (s > bestScore) [best, bestScore] = [c, s];
  }
  return best.id;
}

/**
 * Display titles for the case picker: "MATH-500 " dropped, and titles shared
 * by several cases get the problem number ("Algebra L1 · #2551").
 */
export function caseTitles(cases) {
  const base = (c) => c.title.replace(/^MATH-500\s+/, '');
  const count = new Map();
  for (const c of cases) count.set(base(c), (count.get(base(c)) ?? 0) + 1);
  return new Map(
    cases.map((c) => [c.id, count.get(base(c)) > 1 ? `${base(c)} · #${String(c.id).split('-').pop()}` : base(c)]),
  );
}

/** Short URL form of a case id: "math500-algebra-2551" -> "algebra-2551". */
export function shortCaseId(id) {
  return String(id).replace(/^math500-/, '');
}

/** Find a case by full or short id; null if none matches. */
export function resolveCaseId(cases, query) {
  if (!query) return null;
  const q = String(query).trim();
  const hit = cases.find((c) => c.id === q || shortCaseId(c.id) === q);
  return hit ? hit.id : null;
}

/** Parse "?case=algebra-2551&t=6000&speed=4" against the loaded cases. */
export function parseReplayState(search, cases) {
  const p = new URLSearchParams(search);
  const requested = resolveCaseId(cases, p.get('case'));
  const caseId = requested ?? defaultCaseId(cases);
  const c = cases.find((x) => x.id === caseId);
  const tRaw = p.get('t');
  const t = tRaw == null || tRaw === '' ? NaN : Number(tRaw);
  const speed = Number(p.get('speed'));
  return {
    caseId,
    fromUrl: Boolean(requested) || Number.isFinite(t),
    cursor: Number.isFinite(t) && c ? Math.round(clampCursor(t, replayLength(c))) : null,
    speed: SPEEDS.includes(speed) ? speed : null,
  };
}

/** Serialise replay state into a query string (no leading "?"). */
export function formatReplayState({ caseId, cursor, speed }) {
  const p = new URLSearchParams();
  if (caseId) p.set('case', shortCaseId(caseId));
  if (cursor != null) p.set('t', String(Math.round(cursor)));
  if (speed != null && speed !== DEFAULT_SPEED) p.set('speed', String(speed));
  return p.toString();
}

/* ---------- prompt formatting ---------- */

const INSTRUCTION_RE = /\s*Please reason step by step, and put your final answer within \\boxed\{\}\.?\s*$/;
const SUPERSCRIPT = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹', '+': '⁺', '-': '⁻', x: 'ˣ', n: 'ⁿ' };

function superscript(s) {
  const chars = [...s];
  return chars.every((ch) => SUPERSCRIPT[ch]) ? chars.map((ch) => SUPERSCRIPT[ch]).join('') : `^(${s})`;
}

/** LaTeX-ish math to readable plain text (no renderer, small subset). */
export function formatMath(src) {
  return src
    .replace(/\\\$/g, '$')
    .replace(/\\%/g, '%')
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '$1/$2')
    .replace(/\\(?:left|right)(?![a-zA-Z])/g, '')
    .replace(/\\cdot(?![a-zA-Z])/g, '·')
    .replace(/\\times(?![a-zA-Z])/g, '×')
    .replace(/\\div(?![a-zA-Z])/g, '÷')
    .replace(/\\le(?:q)?(?![a-zA-Z])/g, '≤')
    .replace(/\\ge(?:q)?(?![a-zA-Z])/g, '≥')
    .replace(/\\neq?(?![a-zA-Z])/g, '≠')
    .replace(/\\pi(?![a-zA-Z])/g, 'π')
    .replace(/\\sqrt\{([^{}]*)\}/g, '√($1)')
    .replace(/\\[,;! ]/g, ' ')
    .replace(/\^\{([^{}]*)\}/g, (_, s) => superscript(s))
    .replace(/\^([0-9a-zA-Z+-])/g, (_, s) => superscript(s))
    .replace(/\\([a-zA-Z]+)/g, '$1')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Split a MATH-500 prompt into readable segments: [{ text, math }]. The
 * standard "reason step by step … \boxed{}" suffix is reported separately.
 */
export function formatPrompt(prompt) {
  const hasInstruction = INSTRUCTION_RE.test(prompt);
  const body = prompt.replace(INSTRUCTION_RE, '').replace(/\\\$/g, '\u0000');
  const segments = [];
  const re = /\$([^$]*)\$/g;
  let last = 0;
  let m;
  const plain = (s) => s.replace(/\u0000/g, '$').replace(/\\%/g, '%');
  while ((m = re.exec(body))) {
    if (m.index > last) segments.push({ text: plain(body.slice(last, m.index)), math: false });
    segments.push({ text: formatMath(m[1].replace(/\u0000/g, '\\$')), math: true });
    last = m.index + m[0].length;
  }
  if (last < body.length) segments.push({ text: plain(body.slice(last)), math: false });
  return { segments: segments.filter((s) => s.text), hasInstruction };
}

/** formatPrompt as one plain string. */
export function promptText(prompt) {
  return formatPrompt(prompt).segments.map((s) => s.text).join('');
}

/* ---------- batch stats ---------- */

/**
 * Integer outcome counts for the honest data card from a batchStats block
 * ({ n, base_cap_rate, edited_cap_rate, base_acc, edited_acc }). A run that
 * hit the cap never counts as correct here: correct <= stopped.
 */
export function batchCounts(stats) {
  const n = stats.n;
  const side = (capRate, acc) => {
    const cap = Math.round(capRate * n);
    const stopped = n - cap;
    const correct = Math.min(stopped, Math.round(acc * n));
    return { cap, stopped, correct, stoppedWrong: stopped - correct, capRate, acc };
  };
  return { n, base: side(stats.base_cap_rate, stats.base_acc), edited: side(stats.edited_cap_rate, stats.edited_acc) };
}
