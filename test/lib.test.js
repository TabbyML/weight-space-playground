import { describe, expect, it } from 'vitest';
import {
  clampInt,
  clampFloat,
  fmtBytes,
  niceStep,
  shortName,
  isNearBottom,
  createLoopMeter,
} from '../playground/src/lib.js';

describe('clampInt', () => {
  it('parses and clamps', () => {
    expect(clampInt('4096', 16, 65536, 1)).toBe(4096);
    expect(clampInt('5', 16, 65536, 1)).toBe(16);
    expect(clampInt('999999', 16, 65536, 1)).toBe(65536);
    expect(clampInt('12.9', 0, 100, 1)).toBe(12);
  });
  it('falls back on NaN but keeps 0', () => {
    expect(clampInt('', 0, 10, 7)).toBe(7);
    expect(clampInt('abc', 0, 10, 7)).toBe(7);
    expect(clampInt('0', 0, 10, 7)).toBe(0);
  });
});

describe('clampFloat', () => {
  it('keeps temperature 0 (greedy) instead of falling back', () => {
    expect(clampFloat('0', 0, 2, 0.6)).toBe(0);
    expect(clampFloat('0.0', 0, 2, 0.6)).toBe(0);
  });
  it('falls back on empty / invalid and clamps range', () => {
    expect(clampFloat('', 0, 2, 0.6)).toBe(0.6);
    expect(clampFloat('x', 0, 2, 0.6)).toBe(0.6);
    expect(clampFloat('5', 0, 2, 0.6)).toBe(2);
    expect(clampFloat('-1', 0, 2, 0.6)).toBe(0);
    expect(clampFloat('0.7', 0, 2, 0.6)).toBeCloseTo(0.7);
  });
});

describe('fmtBytes', () => {
  it('formats 1024-based units', () => {
    expect(fmtBytes(0)).toBe('0 B');
    expect(fmtBytes(512)).toBe('512 B');
    expect(fmtBytes(1536)).toBe('1.5 KB');
    expect(fmtBytes(508 * 1024 * 1024)).toBe('508 MB');
    expect(fmtBytes(1.05 * 1024 ** 3)).toBe('1.1 GB');
    expect(fmtBytes(undefined)).toBe('0 B');
  });
});

describe('niceStep', () => {
  it('returns 1/2/5×10^k steps giving about six ticks', () => {
    expect(niceStep(1)).toBe(1);
    expect(niceStep(6)).toBe(1);
    expect(niceStep(12)).toBe(2);
    expect(niceStep(30)).toBe(5);
    expect(niceStep(60)).toBe(10);
    expect(niceStep(4096)).toBe(1000);
    for (const max of [7, 100, 333, 4096, 32768]) {
      const ticks = Math.floor(max / niceStep(max)) + 1;
      expect(ticks).toBeGreaterThanOrEqual(3);
      expect(ticks).toBeLessThanOrEqual(8);
    }
  });
});

describe('shortName', () => {
  it('returns the last URL segment', () => {
    expect(shortName('https://x/y/Qwen3.5-0.8B-Q4_K_M.gguf')).toBe('Qwen3.5-0.8B-Q4_K_M.gguf');
  });
});

describe('isNearBottom', () => {
  it('is true only within the slack of the bottom', () => {
    expect(isNearBottom({ scrollTop: 600, scrollHeight: 1000, clientHeight: 400 })).toBe(true);
    expect(isNearBottom({ scrollTop: 590, scrollHeight: 1000, clientHeight: 400 })).toBe(true);
    expect(isNearBottom({ scrollTop: 100, scrollHeight: 1000, clientHeight: 400 })).toBe(false);
    expect(isNearBottom({ scrollTop: 0, scrollHeight: 300, clientHeight: 400 })).toBe(true);
  });
});

describe('createLoopMeter', () => {
  it('scores non-repeating text as 0', () => {
    const m = createLoopMeter();
    m.push('one two three four five six seven eight nine ten ');
    expect(m.score()).toBe(0);
  });

  it('climbs toward 1 on a repeating loop', () => {
    const m = createLoopMeter({ n: 4, window: 80 });
    const phrase = 'wait let me verify the answer again ';
    for (let i = 0; i < 40; i++) m.push(phrase);
    expect(m.score()).toBeGreaterThan(0.9);
  });

  it('is independent of how the stream is chunked (partial words held back)', () => {
    const text = 'so 2^10 is 1024 and 1024 plus 24 is 1048 so 2^10 is 1024 and 1024 plus 24 is 1048 done';
    const whole = createLoopMeter();
    whole.push(text);
    const a = whole.flush();
    const chunked = createLoopMeter();
    for (let i = 0; i < text.length; i += 3) chunked.push(text.slice(i, i + 3));
    const b = chunked.flush();
    expect(b).toBe(a);
    expect(chunked.wordCount).toBe(whole.wordCount);
    expect(a).toBeGreaterThan(0);
  });

  it('only counts the recent window', () => {
    const m = createLoopMeter({ n: 2, window: 10 });
    for (let i = 0; i < 10; i++) m.push('a b ');
    expect(m.score()).toBeGreaterThan(0.5);
    for (let i = 0; i < 20; i++) m.push(`w${i} `);
    expect(m.score()).toBe(0);
  });
});
