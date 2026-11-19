/**
 * neuron-trace.js  -  pure, DOM-free loading and decoding of the recorded neuron
 * trace (data/neuron-trace/index.json + one <case>.bin per case; schema in
 * playground/public/data/README.md). Unit-tested in test/neuron-trace.test.js.
 *
 * Every per-token series is int8 and every heatmap uint8, each with its own
 * linear scale (value = q * scale). Value i of a series lines up with token i
 * of the same run in replay.json.
 */

export const NEURON_TRACE_DIR = 'data/neuron-trace/';
export const TRACE_RUNS = ['base', 'edited', 'counterfactual'];
export const TRACE_SERIES = ['push', 'proj', 'norm'];
export const HEAT_CHANNELS = 64;

const TYPED = { int8: Int8Array, uint8: Uint8Array };

/** Dequantise one packed array of a case's .bin: Float32Array of q * scale. */
export function decodeArray(buffer, desc) {
  const Typed = TYPED[desc.dtype];
  if (!Typed) throw new Error(`neuron trace: unsupported dtype ${desc.dtype}`);
  if (desc.offset < 0 || desc.offset + desc.length > buffer.byteLength) {
    throw new Error(`neuron trace: array [${desc.offset}, +${desc.length}) outside a ${buffer.byteLength}-byte file`);
  }
  const q = new Typed(buffer, desc.offset, desc.length);
  const out = new Float32Array(desc.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] * desc.scale;
  return out;
}

/**
 * The array descriptor of `key` for (run, layer), following `{ sameAs: run }`
 * links (the counterfactual's layer 15 push / norm / heat equal the base run's).
 */
export function resolveEntry(caseEntry, run, layer, key) {
  const seen = new Set();
  let r = run;
  for (;;) {
    const entry = caseEntry.runs[r]?.layers?.[String(layer)]?.[key];
    if (!entry) throw new Error(`neuron trace: ${caseEntry.id} has no ${r}.${layer}.${key}`);
    if (!entry.sameAs) return { run: r, entry };
    if (seen.has(r)) throw new Error(`neuron trace: sameAs cycle at ${caseEntry.id} ${r}.${layer}.${key}`);
    seen.add(r);
    r = entry.sameAs;
  }
}

/** The index entry of a case id, or null. */
export function findTraceCase(index, id) {
  return index?.cases?.find((c) => c.id === id) ?? null;
}

/**
 * Decode one case: { id, kind, layers, columns, runs: { <run>: { nTokens,
 * binEdges, nBins, layers: { <L>: { push, proj, norm, heat, channels,
 * sameAs } } } } }. `heat` is channel-major [64 * nBins]. `sameAs` names the
 * series that are shared with another run (e.g. { push: 'base' }).
 */
export function decodeTraceCase(index, caseEntry, buffer) {
  const layers = index.layers;
  const columns = Object.fromEntries(layers.map((L) => [L, index.edit[String(L)].columns]));
  const runs = {};
  for (const run of TRACE_RUNS) {
    const meta = caseEntry.runs[run];
    if (!meta) continue;
    const nBins = meta.binEdges.length - 1;
    const out = { nTokens: meta.nTokens, binEdges: meta.binEdges, nBins, layers: {} };
    for (const L of layers) {
      const layer = { sameAs: {} };
      for (const key of [...TRACE_SERIES, 'heat']) {
        const { run: owner, entry } = resolveEntry(caseEntry, run, L, key);
        layer[key] = decodeArray(buffer, entry);
        if (owner !== run) layer.sameAs[key] = owner;
      }
      if (layer.heat.length !== HEAT_CHANNELS * nBins) {
        throw new Error(`neuron trace: ${caseEntry.id} ${run}.${L}.heat has ${layer.heat.length} values, expected 64 × ${nBins}`);
      }
      const ch = resolveEntry(caseEntry, run, L, 'channels').entry;
      layer.channels = ch;
      out.layers[L] = layer;
    }
    runs[run] = out;
  }
  return { id: caseEntry.id, kind: caseEntry.kind, layers, columns, runs };
}

/**
 * Lazy loader: index.json on first use, then each case's .bin only when that
 * case is asked for. Promises are cached, so concurrent calls share a fetch.
 */
export function createTraceLoader({ fetchImpl = (...a) => fetch(...a), dir = NEURON_TRACE_DIR } = {}) {
  let indexPromise = null;
  const cases = new Map();
  const index = () => {
    if (!indexPromise) {
      indexPromise = fetchImpl(`${dir}index.json`).then(async (res) => {
        if (!res.ok) throw new Error(`${dir}index.json: HTTP ${res.status}`);
        const data = await res.json();
        if (data.placeholder || !data.cases?.length) throw new Error('neuron trace: no recorded cases');
        return data;
      });
      indexPromise.catch(() => (indexPromise = null)); // allow a retry later
    }
    return indexPromise;
  };
  const loadCase = (id) => {
    if (!cases.has(id)) {
      const p = index().then(async (idx) => {
        const entry = findTraceCase(idx, id);
        if (!entry) return null; // no trace for this case: the panel says so
        const res = await fetchImpl(`${dir}${entry.file}`);
        if (!res.ok) throw new Error(`${dir}${entry.file}: HTTP ${res.status}`);
        const buffer = await res.arrayBuffer();
        return decodeTraceCase(idx, entry, buffer);
      });
      p.catch(() => cases.delete(id));
      cases.set(id, p);
    }
    return cases.get(id);
  };
  return { index, loadCase };
}

/**
 * Shared y-scale for a layer's push: the largest per-token |push| peak of any
 * base run in the index. Every case is drawn on the same scale, so a control
 * that never loops looks as quiet as it is.
 */
export function pushScale(index, layer) {
  let peak = 0;
  for (const c of index.cases) {
    const entry = c.runs.base?.layers?.[String(layer)]?.push;
    if (entry?.peak > peak) peak = entry.peak;
  }
  return peak || 1;
}

/* ---------- indexing ---------- */

/** Series value at token `i`, or null outside the run. */
export function valueAt(series, i) {
  return i >= 0 && i < series.length ? series[i] : null;
}

/** Heatmap bin holding token `i` (binEdges[b] <= i < binEdges[b + 1]), or -1. */
export function binOf(binEdges, i) {
  const last = binEdges.length - 1;
  if (i < binEdges[0] || i >= binEdges[last]) return -1;
  let lo = 0;
  let hi = last - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (binEdges[mid] <= i) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Downsample a series onto `buckets` equal slices of the token domain
 * [0, domain): bucket b covers [floor(b*domain/buckets), floor((b+1)*domain/buckets)).
 * Returns { mean, min, max } (NaN where the slice is empty or past the run's
 * end). With `abs`, statistics are of |value|.
 */
export function bucketStats(series, domain, buckets, { abs = false } = {}) {
  const mean = new Float32Array(buckets).fill(NaN);
  const min = new Float32Array(buckets).fill(NaN);
  const max = new Float32Array(buckets).fill(NaN);
  for (let b = 0; b < buckets; b++) {
    const from = Math.floor((b * domain) / buckets);
    const to = Math.min(series.length, Math.max(from + 1, Math.floor(((b + 1) * domain) / buckets)));
    if (from >= to) continue;
    let s = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = from; i < to; i++) {
      const v = abs ? Math.abs(series[i]) : series[i];
      s += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    mean[b] = s / (to - from);
    min[b] = lo;
    max[b] = hi;
  }
  return { mean, min, max };
}

/** Trailing moving average of |series| over `window` tokens (inclusive of i). */
export function trailingAbsMean(series, window) {
  const out = new Float32Array(series.length);
  let s = 0;
  for (let i = 0; i < series.length; i++) {
    s += Math.abs(series[i]);
    if (i >= window) s -= Math.abs(series[i - window]);
    out[i] = s / Math.min(window, i + 1);
  }
  return out;
}

/** Hysteresis switch: turns on at >= on, off at < off, otherwise keeps `prev`. */
export function hysteresis(prev, value, { on, off }) {
  if (value == null || Number.isNaN(value)) return false;
  if (prev) return value >= off;
  return value >= on;
}

/**
 * Channels ranked by their share of the run's mean push:
 * [{ k, column, meanPush, share }] sorted by |meanPush| descending, where
 * share = |meanPush_k| / Σ |meanPush|.
 */
export function rankChannels(meanPush, columns) {
  const total = meanPush.reduce((s, v) => s + Math.abs(v), 0) || 1;
  return meanPush
    .map((v, k) => ({ k, column: columns[k], meanPush: v, share: Math.abs(v) / total }))
    .sort((a, b) => Math.abs(b.meanPush) - Math.abs(a.meanPush) || a.k - b.k);
}

/* ---------- verbatim repeats the recorded loop segments miss ---------- */

/**
 * Union of every qualifying periodic span (same rule as
 * tools/precompute/loops.py and neuron_findings.union_loop_mask, but
 * overlaps kept): spans where tokens[i] === tokens[i - p] for a period p in
 * [minPeriod, maxPeriod], covering >= max(minLen, minRepeats * p) tokens
 * (span start = first copied token - p). Returns merged [{ start, end }].
 */
export function verbatimRepeatSpans(tokens, { minPeriod = 4, maxPeriod = 400, minRepeats = 3, minLen = 48 } = {}) {
  const n = tokens.length;
  const mask = new Uint8Array(n);
  const pMax = Math.min(maxPeriod, Math.floor(n / 2));
  for (let p = minPeriod; p <= pMax; p++) {
    const need = Math.max(minLen, minRepeats * p);
    let i = p;
    while (i < n) {
      if (tokens[i] === tokens[i - p]) {
        let j = i;
        while (j < n && tokens[j] === tokens[j - p]) j++;
        if (j - (i - p) >= need) mask.fill(1, i - p, j);
        i = j;
      } else i++;
    }
  }
  const spans = [];
  for (let i = 0; i < n; ) {
    if (!mask[i]) {
      i++;
      continue;
    }
    let j = i;
    while (j < n && mask[j]) j++;
    spans.push({ start: i, end: j });
    i = j;
  }
  return spans;
}

/** Parts of `spans` not covered by any recorded `loops` segment. */
export function subtractSpans(spans, loops) {
  const cuts = [...loops].sort((a, b) => a.start - b.start);
  const out = [];
  for (const s of spans) {
    let from = s.start;
    for (const c of cuts) {
      if (c.end <= from || c.start >= s.end) continue;
      if (c.start > from) out.push({ start: from, end: c.start });
      from = Math.max(from, c.end);
    }
    if (from < s.end) out.push({ start: from, end: s.end });
  }
  return out;
}

/** Total tokens covered by spans. */
export function spanTokens(spans) {
  return spans.reduce((s, x) => s + (x.end - x.start), 0);
}
