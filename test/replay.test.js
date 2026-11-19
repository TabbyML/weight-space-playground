import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  TOKENS_PER_SECOND_1X,
  DEFAULT_SPEED,
  tokensPerSecond,
  cursorToMs,
  msToCursor,
  advanceCursor,
  clampCursor,
  replayLength,
  prepareRun,
  prepareCase,
  textOf,
  repetitionAt,
  loopAt,
  loopTokensAt,
  loopPhrase,
  textSpans,
  runStateAt,
  runVerdict,
  countersAt,
  caseDramaScore,
  defaultCaseId,
  resolveCaseId,
  caseTitles,
  shortCaseId,
  parseReplayState,
  formatReplayState,
  formatMath,
  formatPrompt,
  promptText,
  batchCounts,
} from '../playground/src/replay.js';
import { rank1Summary } from '../playground/src/lib.js';

const replay = JSON.parse(readFileSync(new URL('../playground/public/data/replay.json', import.meta.url), 'utf8'));
const weightDiff = JSON.parse(readFileSync(new URL('../playground/public/data/weight-diff.json', import.meta.url), 'utf8'));

/** A tiny run: 4 intro tokens, then "ab" repeated, then </think> and an answer. */
function fixtureRun() {
  const tokens = ['x', 'y', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', '</think>', '4', '2'];
  return {
    tokens,
    logprobs: tokens.map(() => -0.1),
    thinkEnd: 10,
    finish: 'stop',
    nTokens: tokens.length,
    answer: '42',
    correct: true,
    loops: [{ start: 4, end: 10, period: 2, repeats: 3 }],
  };
}

function fixtureCase(over = {}) {
  const loopRun = (n, loops) => ({
    tokens: Array.from({ length: n }, (_, i) => `t${i} `),
    logprobs: [],
    thinkEnd: null,
    finish: 'length',
    nTokens: n,
    answer: null,
    correct: false,
    loops,
  });
  return {
    id: 'math500-x-1',
    kind: 'loop',
    sampling: { max_tokens: 100 },
    base: loopRun(100, [{ start: 20, end: 100, period: 10, repeats: 8 }]),
    edited: { ...loopRun(40, []), finish: 'stop', thinkEnd: 30, answer: '7', correct: true },
    ...over,
  };
}

describe('clock', () => {
  it('maps speed to tokens per second (1x ≈ 60 tok/s)', () => {
    expect(TOKENS_PER_SECOND_1X).toBe(60);
    expect(tokensPerSecond(1)).toBe(60);
    expect(tokensPerSecond(16)).toBe(960);
  });
  it('converts cursor <-> time and back', () => {
    expect(cursorToMs(60, 1)).toBe(1000);
    expect(cursorToMs(240, 4)).toBe(1000);
    expect(msToCursor(1000, 16)).toBe(960);
    for (const s of [1, 4, 16]) expect(msToCursor(cursorToMs(1234, s), s)).toBeCloseTo(1234, 9);
  });
  it('at the default speed the default case loops and escapes within ~5 s', () => {
    const c = replay.cases.find((x) => x.id === defaultCaseId(replay.cases));
    const firstRepeat = c.base.loops[0].start + c.base.loops[0].period;
    expect(cursorToMs(firstRepeat, DEFAULT_SPEED)).toBeLessThan(5000);
    expect(cursorToMs(c.edited.nTokens, DEFAULT_SPEED)).toBeLessThan(5000);
  });
  it('advances by frame time, clamps to the end, and caps long frames', () => {
    expect(advanceCursor(0, 1000 / 60, 1, 100)).toBeCloseTo(1, 9);
    expect(advanceCursor(99.5, 50, 16, 100)).toBe(100);
    expect(advanceCursor(0, 5000, 1, 1e6)).toBeCloseTo(6, 9); // a 5 s gap counts as 100 ms
    expect(advanceCursor(10, -20, 1, 100)).toBe(10);
  });
  it('clampCursor handles junk', () => {
    expect(clampCursor('abc', 10)).toBe(0);
    expect(clampCursor(-3, 10)).toBe(0);
    expect(clampCursor(30, 10)).toBe(10);
  });
  it('replayLength is the longer run', () => {
    expect(replayLength(fixtureCase())).toBe(100);
  });
});

describe('prepareRun / text', () => {
  const run = prepareRun(fixtureRun());
  it('builds char offsets matching the joined text', () => {
    expect(run.text).toBe(run.tokens.join(''));
    expect(textOf(run, 2, 6)).toBe('abab');
    expect(textOf(run, 0, run.nTokens)).toBe(run.text);
  });
  it('repetition is 0 before any token and in [0,1] after', () => {
    expect(repetitionAt(run, 0)).toBe(0);
    for (let k = 1; k <= run.nTokens; k++) {
      expect(repetitionAt(run, k)).toBeGreaterThanOrEqual(0);
      expect(repetitionAt(run, k)).toBeLessThanOrEqual(1);
    }
  });
  it('repetition rises on a real recorded loop', () => {
    const c = prepareCase(replay.cases.find((x) => x.id === 'math500-algebra-2551'));
    const s = c.base.loops[0];
    expect(repetitionAt(c.base, s.start + 2000)).toBeGreaterThan(0.9);
    expect(repetitionAt(c.base, 200)).toBeLessThan(0.5);
  });
});

describe('loop lookup at a cursor', () => {
  const run = fixtureRun(); // loop [4,10) period 2
  it('is null outside segments', () => {
    expect(loopAt(run, 0)).toBeNull();
    expect(loopAt(run, 4)).toBeNull(); // last written token is 3
    expect(loopAt(run, 11)).toBeNull(); // last written token is 10 (past end)
  });
  it('counts completed repeats and finds the source and current copy', () => {
    const a = loopAt(run, 5);
    expect(a.segment.start).toBe(4);
    expect(a).toMatchObject({ written: 1, repeats: 0, sourceStart: 2, currentStart: 4 });
    expect(loopAt(run, 6)).toMatchObject({ repeats: 1, currentStart: 4 });
    expect(loopAt(run, 7)).toMatchObject({ repeats: 1, currentStart: 6 });
    expect(loopAt(run, 10)).toMatchObject({ repeats: 3, currentStart: 8 });
  });
  it('ends at floor(segment.repeats) on real data', () => {
    const b = replay.cases.find((x) => x.id === 'math500-algebra-2551').base;
    const s = b.loops[0];
    expect(loopAt(b, s.end).repeats).toBe(Math.floor(s.repeats));
  });
  it('loopTokensAt sums written loop tokens', () => {
    expect(loopTokensAt(run, 3)).toBe(0);
    expect(loopTokensAt(run, 7)).toBe(3);
    expect(loopTokensAt(run, 99)).toBe(6);
    const multi = { nTokens: 50, loops: [{ start: 5, end: 10 }, { start: 20, end: 30 }] };
    expect(loopTokensAt(multi, 25)).toBe(10);
  });
  it('loopPhrase picks the longest line of the repeated unit', () => {
    const r = prepareRun({
      ...fixtureRun(),
      tokens: ['\n', '* ', 'short\n', '*   a much longer line here\n', 'x', 'x', 'x', 'x', 'x', 'x', '</think>', '1', '2'],
      loops: [{ start: 4, end: 10, period: 3, repeats: 2 }],
    });
    expect(loopPhrase(r, r.loops[0])).toBe('a much longer line here');
    expect(loopPhrase(r, r.loops[0], 8)).toBe('a much…');
  });
});

describe('textSpans', () => {
  const run = prepareRun(fixtureRun());
  it('marks source, old and current copy of the active loop', () => {
    const spans = textSpans(run, 7);
    expect(spans).toEqual([
      { region: 'think', cls: '', from: 0, to: 2 },
      { region: 'think', cls: 'loop-src', from: 2, to: 4 },
      { region: 'think', cls: 'loop-old', from: 4, to: 6 },
      { region: 'think', cls: 'loop-now', from: 6, to: 7 },
    ]);
  });
  it('drops the </think> token and splits think/answer', () => {
    const spans = textSpans(run, run.nTokens);
    expect(spans.at(-1)).toEqual({ region: 'answer', cls: '', from: 11, to: 13 });
    expect(spans.some((s) => s.from <= 10 && 10 < s.to)).toBe(false);
    const think = spans.filter((s) => s.region === 'think');
    expect(think.map((s) => s.cls)).toEqual(['', 'loop-old']); // finished loop stays tinted, no src/now
  });
  it('is empty at cursor 0 and covers every written token', () => {
    expect(textSpans(run, 0)).toEqual([]);
    const spans = textSpans(prepareRun(replay.cases[0].edited), 6000);
    const covered = spans.reduce((a, s) => a + s.to - s.from, 0);
    expect(covered).toBe(6000 - 1); // minus the </think> token
  });
});

describe('counters at a cursor', () => {
  const c = fixtureCase();
  it('tracks tokens spent, phase and looping waste', () => {
    const k = countersAt(c, 35);
    expect(k.base).toMatchObject({ spent: 35, done: false, phase: 'thinking', loopTokens: 15 });
    expect(k.base.loop.repeats).toBe(1);
    expect(k.edited).toMatchObject({ spent: 35, done: false, phase: 'answer', loopTokens: 0 });
    expect(k.wastedLooping).toBe(15);
    expect(k.verdict).toBeNull();
  });
  it('finishes the shorter run first and gives a verdict at the end', () => {
    const mid = countersAt(c, 60);
    expect(mid.edited).toMatchObject({ spent: 40, done: true, finish: 'stop', answer: '7', correct: true });
    expect(mid.base.done).toBe(false);
    const end = countersAt(c, 1e9);
    expect(end.cursor).toBe(100);
    expect(end.verdict).toEqual({ base: 'hit the 100-token cap · no answer', edited: 'answered 7 ✓ in 40 tokens' });
    expect(end.wastedLooping).toBe(80);
  });
  it('matches the recording for the default case', () => {
    const real = replay.cases.find((x) => x.id === 'math500-algebra-2551');
    const end = countersAt(real, replayLength(real));
    expect(end.base).toMatchObject({ spent: 16384, finish: 'length', correct: false });
    expect(end.edited).toMatchObject({ spent: 7182, finish: 'stop', answer: '10', correct: true });
    expect(end.verdict.edited).toBe('answered 10 ✓ in 7,182 tokens');
    expect(end.verdict.base).toBe('hit the 16,384-token cap · no answer');
    expect(end.base.loopTokens).toBe(16384 - 4733);
  });
  it('runStateAt / runVerdict handle a stop without an answer and a wrong answer', () => {
    const r = { ...fixtureRun(), answer: null };
    expect(runVerdict(r, 100)).toBe('stopped after 13 tokens · no boxed answer');
    expect(runVerdict({ ...fixtureRun(), correct: false }, 100)).toBe('answered 42 ✗ in 13 tokens');
    expect(runStateAt(fixtureRun(), 0)).toMatchObject({ spent: 0, phase: 'thinking', finish: null, answer: null });
  });
});

describe('case default + URL state', () => {
  it('defaults to the most dramatic loop case of the real data', () => {
    expect(defaultCaseId(replay.cases)).toBe('math500-algebra-2551');
  });
  it('scores control and non-qualifying cases 0', () => {
    expect(caseDramaScore(fixtureCase({ kind: 'control' }))).toBe(0);
    expect(caseDramaScore(fixtureCase())).toBeCloseTo(80 / 40);
    expect(defaultCaseId([fixtureCase({ id: 'a', kind: 'control' }), fixtureCase({ id: 'b' })])).toBe('b');
    expect(defaultCaseId([])).toBeNull();
  });
  it('gives duplicate titles the problem number', () => {
    const t = caseTitles(replay.cases);
    expect(t.get('math500-algebra-2551')).toBe('Algebra L1 · #2551');
    expect(t.get('math500-algebra-24')).toBe('Algebra L1 · #24');
    expect(t.get('math500-algebra-346')).toBe('Algebra L2');
    expect(new Set(t.values()).size).toBe(replay.cases.length);
  });
  it('resolves short and full ids', () => {
    expect(shortCaseId('math500-algebra-2551')).toBe('algebra-2551');
    expect(resolveCaseId(replay.cases, 'algebra-2551')).toBe('math500-algebra-2551');
    expect(resolveCaseId(replay.cases, 'math500-algebra-24')).toBe('math500-algebra-24');
    expect(resolveCaseId(replay.cases, 'nope')).toBeNull();
  });
  it('parses and formats the query string', () => {
    const s = parseReplayState('?case=algebra-2551&t=6000&speed=4', replay.cases);
    expect(s).toEqual({ caseId: 'math500-algebra-2551', fromUrl: true, cursor: 6000, speed: 4 });
    expect(parseReplayState('', replay.cases)).toEqual({ caseId: 'math500-algebra-2551', fromUrl: false, cursor: null, speed: null });
    expect(parseReplayState('?case=prealgebra-192&t=99999&speed=3', replay.cases)).toMatchObject({ cursor: 4584, speed: null });
    expect(formatReplayState({ caseId: 'math500-algebra-2551', cursor: 6000.4, speed: DEFAULT_SPEED })).toBe('case=algebra-2551&t=6000');
    expect(parseReplayState('?speed=32', replay.cases).speed).toBe(32);
    expect(formatReplayState({ caseId: 'math500-algebra-24', cursor: null, speed: 1 })).toBe('case=algebra-24&speed=1');
  });
});

describe('prompt formatting', () => {
  it('renders the LaTeX-ish prompts readably', () => {
    expect(promptText(replay.cases[0].prompt)).toBe(
      'One ticket to a show costs $20 at full price. Susan buys 4 tickets using a coupon that gives her a 25% discount. Pam buys 5 tickets using a coupon that gives her a 30% discount. How many more dollars does Pam pay than Susan?',
    );
    expect(promptText('For what value of $x$ is $2^3\\cdot3^x=72$? Please reason step by step, and put your final answer within \\boxed{}.')).toBe(
      'For what value of x is 2³·3ˣ=72?',
    );
    expect(formatPrompt(replay.cases[0].prompt).hasInstruction).toBe(true);
  });
  it('handles fractions, braces and unknown commands', () => {
    expect(formatMath('\\frac{1}{2} + x^{10} - \\alpha')).toBe('1/2 + x¹⁰ - alpha');
    expect(formatMath('g(f(5)-1)')).toBe('g(f(5)-1)');
    expect(formatMath('2^{ab}')).toBe('2^(ab)');
  });
  it('marks math segments', () => {
    const { segments } = formatPrompt('Let $f(x) = 2x-3$ and go.');
    expect(segments).toEqual([
      { text: 'Let ', math: false },
      { text: 'f(x) = 2x-3', math: true },
      { text: ' and go.', math: false },
    ]);
  });
});

describe('batch counts', () => {
  it('derives the honest card numbers from batchStats', () => {
    const k = batchCounts(replay.batchStats);
    expect(k.n).toBe(31);
    expect(k.base).toMatchObject({ cap: 31, stopped: 0, correct: 0 });
    expect(k.edited).toMatchObject({ cap: 17, stopped: 14, correct: 10, stoppedWrong: 4 });
  });
});

describe('rank1Summary', () => {
  it('summarises the real bf16 edit', () => {
    const s = rank1Summary(weightDiff);
    expect(s.tensors).toBe(2);
    expect(s.columnsPerTensor).toBe(64);
    expect(Math.round(s.alpha)).toBe(6);
    expect(s.minRank1Energy).toBeGreaterThan(0.9999);
  });
  it('is null without SVD data', () => {
    expect(rank1Summary({ tensors: [{ changedColumns: [1] }] })).toBeNull();
    expect(rank1Summary(null)).toBeNull();
  });
});
