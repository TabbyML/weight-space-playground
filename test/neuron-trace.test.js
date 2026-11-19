import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  decodeArray,
  resolveEntry,
  findTraceCase,
  decodeTraceCase,
  createTraceLoader,
  valueAt,
  binOf,
  bucketStats,
  trailingAbsMean,
  hysteresis,
  rankChannels,
  verbatimRepeatSpans,
  subtractSpans,
  spanTokens,
  pushScale,
} from '../playground/src/neuron-trace.js';

/**
 * Tiny synthetic neuron-trace fixture in the served format: arrays packed back
 * to back into one buffer, each described by { offset, length, dtype, scale }.
 */
function buildFixture() {
  const chunks = [];
  let offset = 0;
  const pack = (dtype, q, scale) => {
    const bytes = dtype === 'int8' ? new Int8Array(q) : new Uint8Array(q);
    chunks.push(new Uint8Array(bytes.buffer));
    const desc = { offset, length: q.length, dtype, shape: [q.length], scale, peak: 127 * scale, mean: 0 };
    offset += q.length;
    return desc;
  };
  const heat = (nBins, fill) => pack('uint8', Array.from({ length: 64 * nBins }, (_, i) => (i + fill) % 256), 0.01);
  const meanPush = Array.from({ length: 64 }, (_, k) => (k === 3 ? -2 : k === 7 ? 1 : 0.01));
  const channels = { meanAbs: Array(64).fill(0.1), meanPush };
  const layer = (n, base, nBins) => ({
    push: pack('int8', Array.from({ length: n }, (_, i) => base + i), 0.5),
    proj: pack('int8', Array.from({ length: n }, (_, i) => -(base + i)), 0.25),
    norm: pack('int8', Array.from({ length: n }, () => 100), 0.04),
    heat: heat(nBins, base),
    channels,
  });
  const nBase = 5;
  const nEdited = 3;
  const runs = {
    base: { weights: 'base', tokens: 'base', nTokens: nBase, binEdges: [0, 1, 2, 3, 4, 5], layers: { 15: layer(nBase, 10, 5), 19: layer(nBase, -20, 5) } },
    edited: { weights: 'edited', tokens: 'edited', nTokens: nEdited, binEdges: [0, 1, 2, 3], layers: { 15: layer(nEdited, 1, 3), 19: layer(nEdited, 2, 3) } },
    counterfactual: {
      weights: 'edited',
      tokens: 'base',
      nTokens: nBase,
      binEdges: [0, 1, 2, 3, 4, 5],
      layers: {
        15: {
          push: { sameAs: 'base' },
          norm: { sameAs: 'base' },
          heat: { sameAs: 'base' },
          channels: { sameAs: 'base' },
          proj: pack('int8', [1, 2, 3, 4, 5], 1),
        },
        19: layer(nBase, 30, 5),
      },
    },
  };
  const buf = new Uint8Array(offset);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  const columns = Array.from({ length: 64 }, (_, k) => k * 10);
  const index = {
    placeholder: false,
    formatVersion: 1,
    layers: [15, 19],
    edit: { 15: { columns, pushWeights: [] }, 19: { columns: columns.map((c) => c + 1), pushWeights: [] } },
    cases: [{ id: 'case-a', kind: 'loop', file: 'case-a.bin', bytes: offset, runs }],
  };
  return { index, buffer: buf.buffer };
}

describe('decodeArray', () => {
  it('dequantises int8 and uint8 with the per-array scale', () => {
    const buf = new Uint8Array([0xff, 0x80, 0x7f, 200]).buffer; // int8: -1, -128, 127
    expect(Array.from(decodeArray(buf, { offset: 0, length: 3, dtype: 'int8', scale: 0.5 }))).toEqual([-0.5, -64, 63.5]);
    expect(Array.from(decodeArray(buf, { offset: 3, length: 1, dtype: 'uint8', scale: 0.1 }))[0]).toBeCloseTo(20, 5);
  });

  it('rejects arrays outside the file and unknown dtypes', () => {
    const buf = new ArrayBuffer(4);
    expect(() => decodeArray(buf, { offset: 2, length: 3, dtype: 'int8', scale: 1 })).toThrow(/outside/);
    expect(() => decodeArray(buf, { offset: 0, length: 1, dtype: 'float32', scale: 1 })).toThrow(/dtype/);
  });
});

describe('resolveEntry / findTraceCase', () => {
  const { index } = buildFixture();
  const c = findTraceCase(index, 'case-a');

  it('finds cases by id', () => {
    expect(c.id).toBe('case-a');
    expect(findTraceCase(index, 'nope')).toBeNull();
  });

  it('follows sameAs to the owning run', () => {
    expect(resolveEntry(c, 'counterfactual', 15, 'push').run).toBe('base');
    expect(resolveEntry(c, 'counterfactual', 15, 'proj').run).toBe('counterfactual');
    expect(resolveEntry(c, 'counterfactual', 19, 'push').run).toBe('counterfactual');
    expect(() => resolveEntry(c, 'base', 15, 'missing')).toThrow(/no base\.15\.missing/);
  });
});

describe('decodeTraceCase', () => {
  const { index, buffer } = buildFixture();
  const t = decodeTraceCase(index, index.cases[0], buffer);

  it('decodes every run, layer and series with its own scale', () => {
    expect(Object.keys(t.runs)).toEqual(['base', 'edited', 'counterfactual']);
    expect(Array.from(t.runs.base.layers[15].push)).toEqual([5, 5.5, 6, 6.5, 7]);
    expect(Array.from(t.runs.base.layers[15].proj)).toEqual([-2.5, -2.75, -3, -3.25, -3.5]);
    expect(t.runs.base.layers[19].push[0]).toBe(-10);
    expect(t.runs.edited.layers[19].push.length).toBe(3);
    expect(t.runs.base.layers[15].norm[0]).toBeCloseTo(4, 5);
    expect(t.columns[19][0]).toBe(1);
  });

  it('resolves the counterfactual layer-15 sameAs series to the base run', () => {
    const cf = t.runs.counterfactual.layers[15];
    expect(Array.from(cf.push)).toEqual(Array.from(t.runs.base.layers[15].push));
    expect(cf.sameAs).toEqual({ push: 'base', norm: 'base', heat: 'base' });
    expect(cf.channels.meanPush[3]).toBe(-2);
    expect(Array.from(cf.proj)).toEqual([1, 2, 3, 4, 5]);
    expect(t.runs.counterfactual.layers[19].sameAs).toEqual({});
  });

  it('keeps the heatmap channel-major [64, nBins]', () => {
    const h = t.runs.edited.layers[15];
    expect(t.runs.edited.nBins).toBe(3);
    expect(h.heat.length).toBe(64 * 3);
    // channel k, bin b is element k * nBins + b; the fixture writes (i + 1) % 256
    expect(h.heat[2 * 3 + 1]).toBeCloseTo(((2 * 3 + 1 + 1) % 256) * 0.01, 5);
  });

  it('rejects a heatmap of the wrong shape', () => {
    const bad = structuredClone(index.cases[0]);
    bad.runs.edited.binEdges = [0, 1, 3];
    expect(() => decodeTraceCase(index, bad, buffer)).toThrow(/expected 64/);
  });
});

describe('pushScale', () => {
  it('is the largest base-run push peak of the layer across cases', () => {
    const { index } = buildFixture();
    index.cases.push(structuredClone(index.cases[0]));
    index.cases[1].runs.base.layers[19].push.peak = 99;
    expect(pushScale(index, 19)).toBe(99);
    expect(pushScale(index, 15)).toBe(127 * 0.5);
    expect(pushScale({ cases: [] }, 19)).toBe(1);
  });
});

describe('createTraceLoader', () => {
  const { index, buffer } = buildFixture();
  const fakeFetch = () => {
    const calls = [];
    const impl = async (url) => {
      calls.push(url);
      if (url.endsWith('index.json')) return { ok: true, json: async () => index };
      if (url.endsWith('case-a.bin')) return { ok: true, arrayBuffer: async () => buffer };
      return { ok: false, status: 404 };
    };
    return { calls, impl };
  };

  it('fetches the index once and a case binary only when that case is asked for', async () => {
    const f = fakeFetch();
    const loader = createTraceLoader({ fetchImpl: f.impl, dir: 'nt/' });
    expect(f.calls).toEqual([]);
    const [a, b] = await Promise.all([loader.loadCase('case-a'), loader.loadCase('case-a')]);
    expect(a).toBe(b);
    expect(a.runs.base.nTokens).toBe(5);
    expect(f.calls).toEqual(['nt/index.json', 'nt/case-a.bin']);
    expect(await loader.loadCase('unknown')).toBeNull();
    expect(f.calls).toEqual(['nt/index.json', 'nt/case-a.bin']);
  });

  it('surfaces HTTP errors and allows a retry', async () => {
    let fail = true;
    const loader = createTraceLoader({
      dir: 'nt/',
      fetchImpl: async (url) => (fail ? { ok: false, status: 503 } : url.endsWith('index.json') ? { ok: true, json: async () => index } : { ok: true, arrayBuffer: async () => buffer }),
    });
    await expect(loader.loadCase('case-a')).rejects.toThrow(/503/);
    fail = false;
    expect((await loader.loadCase('case-a')).id).toBe('case-a');
  });
});

describe('indexing helpers', () => {
  it('valueAt is null outside the run', () => {
    const s = Float32Array.of(1, 2, 3);
    expect(valueAt(s, 0)).toBe(1);
    expect(valueAt(s, 3)).toBeNull();
    expect(valueAt(s, -1)).toBeNull();
  });

  it('binOf finds the bin holding a token', () => {
    const edges = [0, 64, 128, 192, 200];
    expect(binOf(edges, 0)).toBe(0);
    expect(binOf(edges, 63)).toBe(0);
    expect(binOf(edges, 64)).toBe(1);
    expect(binOf(edges, 199)).toBe(3);
    expect(binOf(edges, 200)).toBe(-1);
    expect(binOf(edges, -1)).toBe(-1);
  });

  it('bucketStats downsamples onto a token domain, NaN past the run end', () => {
    const s = Float32Array.of(1, -3, 2, 4, -1, 5);
    const st = bucketStats(s, 8, 4);
    expect(Array.from(st.mean.slice(0, 3))).toEqual([-1, 3, 2]);
    expect(Array.from(st.min.slice(0, 3))).toEqual([-3, 2, -1]);
    expect(Array.from(st.max.slice(0, 3))).toEqual([1, 4, 5]);
    expect(Number.isNaN(st.mean[3])).toBe(true);
    expect(bucketStats(s, 6, 2, { abs: true }).mean[0]).toBe(2);
    // more buckets than tokens: every bucket still covers one token
    expect(Array.from(bucketStats(Float32Array.of(7, 9), 2, 4).mean)).toEqual([7, 7, 9, 9]);
  });

  it('trailingAbsMean averages |x| over a trailing window', () => {
    expect(Array.from(trailingAbsMean(Float32Array.of(-2, 4, -6, 8), 2))).toEqual([2, 3, 5, 7]);
  });

  it('hysteresis switches on high, off low, holds in between', () => {
    const o = { on: 3, off: 2 };
    expect(hysteresis(false, 2.5, o)).toBe(false);
    expect(hysteresis(false, 3, o)).toBe(true);
    expect(hysteresis(true, 2.5, o)).toBe(true);
    expect(hysteresis(true, 1.9, o)).toBe(false);
    expect(hysteresis(true, null, o)).toBe(false);
  });

  it('rankChannels orders by share of |mean push|, ties by channel', () => {
    const r = rankChannels([0.5, -2, 0.5, 1], [10, 11, 12, 13]);
    expect(r.map((x) => x.column)).toEqual([11, 13, 10, 12]);
    expect(r[0].share).toBeCloseTo(0.5, 6);
    expect(r[0].k).toBe(1);
  });

});

describe('verbatimRepeatSpans', () => {
  it('finds a periodic span covering >= 3 periods and >= minLen tokens', () => {
    const toks = ['x', 'y', ...Array.from({ length: 12 }, (_, i) => 'abc'[i % 3]), 'z'];
    expect(verbatimRepeatSpans(toks, { minPeriod: 3, minLen: 9 })).toEqual([{ start: 2, end: 14 }]);
    expect(verbatimRepeatSpans(toks, { minPeriod: 3, minLen: 13 })).toEqual([]);
  });

  it('keeps overlapping spans that a non-overlapping pick would drop', () => {
    // period-4 run then period-5 run that overlap at the seam
    const a = Array.from({ length: 16 }, (_, i) => 'abcd'[i % 4]);
    const b = Array.from({ length: 20 }, (_, i) => 'cdxyz'[i % 5]);
    const toks = [...a, ...b];
    const spans = verbatimRepeatSpans(toks, { minPeriod: 4, minLen: 12 });
    expect(spanTokens(spans)).toBe(36);
  });

  it('subtractSpans keeps only what the recorded segments miss', () => {
    expect(subtractSpans([{ start: 0, end: 100 }], [{ start: 20, end: 30 }, { start: 60, end: 120 }])).toEqual([
      { start: 0, end: 20 },
      { start: 30, end: 60 },
    ]);
  });

  it('matches the findings on the real recording: algebra-346 base = 10,096 repeat tokens vs 5,362 marked', () => {
    const replay = JSON.parse(readFileSync(new URL('../playground/public/data/replay.json', import.meta.url), 'utf8'));
    const c = replay.cases.find((x) => x.id === 'math500-algebra-346');
    const union = verbatimRepeatSpans(c.base.tokens);
    expect(spanTokens(union)).toBe(10096);
    expect(spanTokens(c.base.loops)).toBe(5362);
    const extra = subtractSpans(union, c.base.loops);
    expect(extra[0].start).toBeGreaterThan(5000);
    expect(extra[0].start).toBeLessThan(7000);
  });
});

describe('served neuron-trace data', () => {
  it('decodes every real case, and layer 19 base ranks a channel from the data', async () => {
    const dir = new URL('../playground/public/data/neuron-trace/', import.meta.url);
    const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8'));
    for (const entry of index.cases) {
      const bytes = readFileSync(new URL(entry.file, dir));
      const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      const t = decodeTraceCase(index, entry, buffer);
      const push = t.runs.base.layers[19].push;
      const meta = entry.runs.base.layers['19'].push;
      expect(push.length).toBe(entry.runs.base.nTokens);
      expect(Math.abs(push.reduce((a, v) => a + v, 0) / push.length - meta.mean)).toBeLessThanOrEqual(meta.scale / 2 + 1e-6);
      const top = rankChannels(t.runs.base.layers[19].channels.meanPush, t.columns[19])[0];
      expect(t.columns[19]).toContain(top.column);
    }
  });
});
