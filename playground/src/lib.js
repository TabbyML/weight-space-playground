/**
 * lib.js  -  pure, DOM-free helpers shared by main.js and viz.js (unit-tested in
 * test/lib.test.js).
 */

/** Parse an integer setting; NaN falls back to `def`, otherwise clamped to [min, max]. */
export function clampInt(v, min, max, def) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) return def;
  return Math.min(max, Math.max(min, n));
}

/**
 * Parse a float setting; NaN falls back to `def`, otherwise clamped to [min, max].
 * Unlike `parseFloat(v) || def`, a valid 0 is kept (temperature 0 = greedy).
 */
export function clampFloat(v, min, max, def) {
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return def;
  return Math.min(max, Math.max(min, n));
}

/** Human-readable byte size (1024-based): "508 MB", "1.0 GB". */
export function fmtBytes(n) {
  n = Number(n) || 0;
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return (i === 0 ? n.toFixed(0) : n.toFixed(n < 10 ? 1 : 0)) + ' ' + u[i];
}

/** Axis tick step giving roughly six 1/2/5×10^k ticks over [0, max]. */
export function niceStep(max) {
  const raw = max / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw))));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  return Math.max(1, Math.round(step));
}

/** Last path segment of a URL (the GGUF file name). */
export function shortName(url) {
  return String(url).split('/').pop() ?? '';
}

/** True when a scroll box is within `slack` px of its bottom (auto-scroll only then). */
export function isNearBottom({ scrollTop, scrollHeight, clientHeight }, slack = 24) {
  return scrollHeight - scrollTop - clientHeight <= slack;
}

/**
 * Incremental repetition ("loop") metric over streamed text.
 *
 * Score = fraction of the last `window` words whose preceding `n`-gram was
 * already seen earlier in the stream. Text is fed chunk by chunk; a trailing
 * partial word is held back until whitespace (or `flush()`) completes it, so
 * the cost per chunk is proportional to the chunk, not the whole text.
 */
export function createLoopMeter({ n = 4, window = 80 } = {}) {
  const seen = new Set();
  const words = [];
  const flags = [];
  let carry = '';

  const addWord = (word) => {
    words.push(word);
    const i = words.length - 1;
    let repeat = 0;
    if (i >= n) {
      const gram = words.slice(i - n, i).join(' ');
      repeat = seen.has(gram) ? 1 : 0;
      seen.add(gram);
    }
    flags.push(repeat);
  };

  const score = () => {
    const recent = flags.slice(-window);
    return recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  };

  return {
    push(text) {
      carry += text;
      const parts = carry.split(/\s+/);
      carry = parts.pop() ?? '';
      for (const w of parts) if (w) addWord(w);
      return score();
    },
    flush() {
      if (carry) addWord(carry);
      carry = '';
      return score();
    },
    score,
    get wordCount() {
      return words.length;
    },
  };
}

/**
 * Rank-1 summary of a weight-diff.json (schema in public/data/README.md): the
 * edit's scale alpha (mean over tensors), neurons per tensor, and the weakest
 * rank-1 energy fraction. Null when the data has no SVD/alpha fields.
 */
export function rank1Summary(weightDiff) {
  const tensors = weightDiff?.tensors ?? [];
  if (!tensors.length || tensors.some((t) => t.alphaMean == null || !t.svd)) return null;
  const alpha = tensors.reduce((a, t) => a + t.alphaMean, 0) / tensors.length;
  const cols = [...new Set(tensors.map((t) => t.changedColumns.length))];
  return {
    tensors: tensors.length,
    alpha,
    columnsPerTensor: cols.length === 1 ? cols[0] : null,
    minRank1Energy: Math.min(...tensors.map((t) => t.svd.rank1EnergyFraction)),
  };
}
