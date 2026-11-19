/**
 * neuron-view.js  -  the recorded neuron trace, synced to the replay cursor.
 *
 * Two views of the same data (decoded by neuron-trace.js):
 *  - a compact "neuron push" lane under the replay timeline (layer 19, base
 *    coral vs edited mint), so the loop moment is visible while it plays;
 *  - the full "Key neurons during inference" panel (#neurons): per-token push
 *    chart for base / edited / counterfactual, a 64-channel heatmap strip, and
 *    a readout of the token under the cursor or the pointer.
 *
 * Both are canvases with fixed CSS heights from first paint. Everything that
 * depends only on (case, layer, run, size) is drawn once into an offscreen
 * layer; a frame only blits it and draws the cursor, so scrubbing 16k tokens
 * stays cheap and nothing reflows.
 */
import { COLORS, FONT, SANS, DISPLAY, roundRect, timelineGeometry } from './viz.js';
import {
  createTraceLoader,
  valueAt,
  bucketStats,
  trailingAbsMean,
  hysteresis,
  rankChannels,
  pushScale,
  verbatimRepeatSpans,
  subtractSpans,
} from './neuron-trace.js';
import { replayLength, capOf } from './replay.js';

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
const signed = (v) => (v == null ? ' - ' : `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(2)}`);
const LANE_LAYER = 19; // clearest base-vs-edited contrast (findings §1)
const DEFAULT_LAYER = 19;
/** Base-run layer-19 |push| (trailing mean) that makes the mascot react. */
const SPIKE = { window: 96, on: 2.6, off: 2.0 };
const CF = '#8c6bff';

/** Size a canvas (and its offscreen layer) to its CSS box × DPR; returns the CSS size. */
function measure(canvas, layer) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  for (const c of [canvas, layer]) {
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
  }
  return { w, h, dpr };
}

function layerCtx(layer, dpr) {
  const ctx = layer.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, layer.width, layer.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Blit the static layer, then fade the unplayed part and draw the playhead. */
function paintFrame(canvas, layer, size, { x0, x1, top, bottom, cursorX, hoverX, dot }) {
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(layer, 0, 0);
  ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
  if (cursorX == null) return;
  const cx = Math.max(x0, Math.min(x1, cursorX));
  ctx.fillStyle = 'rgba(255, 250, 241, 0.62)';
  ctx.fillRect(cx, top, x1 - cx, bottom - top);
  if (hoverX != null) {
    ctx.strokeStyle = COLORS.dim;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hoverX, top);
    ctx.lineTo(hoverX, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx, top);
  ctx.lineTo(cx, bottom);
  ctx.stroke();
  if (dot) {
    ctx.fillStyle = COLORS.sun;
    ctx.beginPath();
    ctx.arc(cx, bottom, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

/** Loop shading: recorded segments filled coral, missed repeats hatched. */
function shadeLoops(ctx, xOf, top, bottom, loops, extra) {
  ctx.fillStyle = 'rgba(255, 111, 89, 0.16)';
  for (const s of loops) ctx.fillRect(xOf(s.start), top, Math.max(1.5, xOf(s.end) - xOf(s.start)), bottom - top);
  for (const s of extra) {
    const x = xOf(s.start);
    const w = Math.max(1.5, xOf(s.end) - x);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, top, w, bottom - top);
    ctx.clip();
    ctx.strokeStyle = 'rgba(255, 111, 89, 0.38)';
    ctx.lineWidth = 1.5;
    const hgt = bottom - top;
    for (let i = x - hgt; i < x + w; i += 8) {
      ctx.beginPath();
      ctx.moveTo(i, bottom);
      ctx.lineTo(i + hgt, top);
      ctx.stroke();
    }
    ctx.restore();
  }
}

/** Mean line of downsampled stats (NaN = gap), one bucket per `step` CSS px. */
function strokeStats(ctx, stats, x0, yOf, { color, width = 2.5, dash = null, step = 1 }) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.setLineDash(dash ?? []);
  ctx.beginPath();
  let pen = false;
  for (let b = 0; b < stats.mean.length; b++) {
    const v = stats.mean[b];
    if (Number.isNaN(v)) {
      pen = false;
      continue;
    }
    const x = x0 + (b + 0.5) * step;
    pen ? ctx.lineTo(x, yOf(v)) : ctx.moveTo(x, yOf(v));
    pen = true;
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Heatmap ramp: paper -> grape (the edit colour), sqrt for contrast. */
function heatRGB(t) {
  const v = Math.sqrt(Math.max(0, Math.min(1, t)));
  return [Math.round(255 + v * (92 - 255)), Math.round(250 + v * (56 - 250)), Math.round(241 + v * (230 - 241))];
}

function tokenLabel(piece) {
  if (piece == null) return '';
  if (piece === '') return '“” (byte of the next char)';
  const shown = piece.replace(/\n/g, '⏎').replace(/\t/g, '⇥');
  return `“${shown.length > 28 ? shown.slice(0, 27) + '…' : shown}”`;
}

/**
 * Create the neuron views. `seek(cursor)` moves the shared replay cursor;
 * `onLoad()` runs when a case's trace arrives (the replay re-renders so the
 * mascot can react while paused). Call update(case, cursor) on every replay
 * render; resize() on layout change.
 */
export function createNeuronView({ seek, onLoad }) {
  const loader = createTraceLoader();
  const lane = { canvas: $('rp-neuron-lane'), layer: document.createElement('canvas'), key: '' };
  const chart = { canvas: $('nt-chart'), layer: document.createElement('canvas'), key: '' };
  const heat = { canvas: $('nt-heat'), layer: document.createElement('canvas'), key: '' };
  const els = {
    badge: $('nt-badge'),
    caseLabel: $('nt-case'),
    top: $('nt-top'),
    tokenHead: $('nt-token'),
    rows: {
      base: $('nt-row-base'),
      edited: $('nt-row-edited'),
      counterfactual: $('nt-row-counterfactual'),
    },
    scrub: $('nt-scrub'),
    sameNote: $('nt-same'),
  };

  const state = {
    c: null,
    cursor: 0,
    layer: DEFAULT_LAYER,
    heatRun: 'base',
    hover: null, // token index under the pointer, or null
    index: null, // neuron-trace/index.json once loaded
    trace: null, // decoded trace of state.c, or null while loading
    status: 'idle', // idle | loading | ready | missing | error
    error: '',
    spike: false,
    panelVisible: true,
    sizeGen: 0,
  };
  const perCase = new Map(); // id -> { extra, smooth }

  function caseExtras(c) {
    if (!perCase.has(c.id)) {
      const union = verbatimRepeatSpans(c.base.tokens);
      perCase.set(c.id, { extra: subtractSpans(union, c.base.loops ?? []), smooth: null });
    }
    return perCase.get(c.id);
  }

  function load(c) {
    state.trace = null;
    state.status = 'loading';
    state.error = '';
    els.badge.hidden = false;
    els.badge.textContent = 'loading…';
    els.badge.className = 'badge recording';
    Promise.all([loader.index(), loader.loadCase(c.id)]).then(
      ([index, trace]) => {
        if (state.c?.id !== c.id) return;
        state.index = index;
        state.trace = trace;
        state.status = trace ? 'ready' : 'missing';
        if (trace) {
          const x = caseExtras(c);
          x.smooth ??= trailingAbsMean(trace.runs.base.layers[LANE_LAYER].push, SPIKE.window);
        }
        if (trace) {
          els.badge.hidden = true;
        } else {
          els.badge.hidden = false;
          els.badge.textContent = 'no trace for this case';
          els.badge.className = 'badge recording';
        }
        invalidate();
        frame();
        onLoad?.();
      },
      (err) => {
        if (state.c?.id !== c.id) return;
        console.error(err);
        state.status = 'error';
        state.error = err instanceof Error ? err.message : String(err);
        els.badge.hidden = false;
        els.badge.textContent = 'not loaded';
        els.badge.className = 'badge recording';
        invalidate();
        frame();
      },
    );
  }

  /** Measured size of a view, re-measured only after resize() (no per-frame layout reads). */
  const sizeOf = (v) => {
    if (v.sizeGen !== state.sizeGen || !v.size) {
      v.size = measure(v.canvas, v.layer);
      v.sizeGen = state.sizeGen;
    }
    return v.size;
  };

  const invalidate = () => {
    lane.key = '';
    chart.key = '';
    heat.key = '';
  };

  /* ---------- geometry ---------- */
  const panelPad = { l: 52, r: 14 };
  const panelX = (w, domain) => {
    const plotW = Math.max(1, Math.floor(w - panelPad.l - panelPad.r));
    return { x0: panelPad.l, plotW, xOf: (i) => panelPad.l + (i / domain) * plotW, domain };
  };
  const tokenFromX = (canvas, clientX) => {
    const rect = canvas.getBoundingClientRect();
    const { x0, plotW, domain } = panelX(rect.width, replayLength(state.c));
    return Math.max(0, Math.min(domain - 1, Math.floor(((clientX - rect.left - x0) / plotW) * domain)));
  };

  /* ---------- replay lane ---------- */
  function drawLaneStatic(size) {
    const c = state.c;
    const ctx = layerCtx(lane.layer, size.dpr);
    const { w, h } = size;
    const { padL, plotW } = timelineGeometry(w);
    const maxN = Math.max(replayLength(c), capOf(c) ?? 0);
    const xOf = (i) => padL + (i / maxN) * plotW;
    const top = 4;
    const bottom = h - 4;
    ctx.fillStyle = COLORS.ink;
    ctx.font = DISPLAY(13);
    ctx.fillText('neurons', 6, h / 2 - 1);
    ctx.fillStyle = COLORS.dim;
    ctx.font = FONT(9.5);
    ctx.fillText(`L${LANE_LAYER} push`, 6, h / 2 + 11);
    if (state.status !== 'ready') {
      ctx.strokeStyle = COLORS.dim2;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      roundRect(ctx, padL + 1, top + 2, plotW - 2, bottom - top - 4, (bottom - top - 4) / 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.dim;
      ctx.font = SANS(12);
      const msg =
        state.status === 'loading'
          ? 'loading the neuron trace…'
          : state.status === 'missing'
            ? 'no neuron trace for this case'
            : `neuron trace not loaded (${state.error})`;
      ctx.fillText(msg, padL + 14, h / 2 + 4);
      return { x0: padL, x1: padL + plotW, xOf, top, bottom };
    }
    const t = state.trace;
    shadeLoops(ctx, xOf, top, bottom, c.base.loops ?? [], caseExtras(c).extra);
    const runs = [
      { run: t.runs.base, color: COLORS.base, fill: 'rgba(255, 111, 89, 0.35)' },
      { run: t.runs.edited, color: COLORS.edited, fill: 'rgba(31, 181, 137, 0.30)' },
    ];
    const ymax = pushScale(state.index, LANE_LAYER);
    const stats = runs.map(({ run }) => bucketStats(run.layers[LANE_LAYER].push, maxN, Math.floor(plotW / 2), { abs: true }));
    const yOf = (v) => bottom - (Math.min(v, ymax) / ymax) * (bottom - top - 2);
    runs.forEach((r, i) => {
      // soft area under the line, then the line
      const s = stats[i];
      ctx.fillStyle = r.fill;
      for (let b = 0; b < s.mean.length; b++) {
        if (Number.isNaN(s.mean[b])) continue;
        const y = yOf(s.mean[b]);
        ctx.fillRect(padL + b * 2, y, 2, bottom - y);
      }
      strokeStats(ctx, s, padL, yOf, { color: r.color, width: 2, step: 2 });
    });
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, bottom + 0.5);
    ctx.lineTo(padL + plotW, bottom + 0.5);
    ctx.stroke();
    return { x0: padL, x1: padL + plotW, xOf, top, bottom };
  }

  function drawLane() {
    if (!lane.canvas || !state.c) return;
    const size = sizeOf(lane);
    const key = `${state.c.id}|${state.status}|${size.w}x${size.h}@${size.dpr}`;
    if (lane.key !== key) {
      lane.geo = drawLaneStatic(size);
      lane.key = key;
    }
    const g = lane.geo;
    paintFrame(lane.canvas, lane.layer, size, {
      x0: g.x0,
      x1: g.x1,
      top: 0,
      bottom: size.h,
      cursorX: g.xOf(state.cursor),
    });
  }

  /* ---------- panel: push chart ---------- */
  function drawChartStatic(size) {
    const c = state.c;
    const ctx = layerCtx(chart.layer, size.dpr);
    const { w, h } = size;
    const domain = replayLength(c);
    const { x0, plotW, xOf } = panelX(w, domain);
    const top = 10;
    const bottom = h - 22;
    const geo = { x0, x1: x0 + plotW, xOf, top, bottom };

    // x axis + ticks
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, top);
    ctx.lineTo(x0, bottom);
    ctx.lineTo(x0 + plotW, bottom);
    ctx.stroke();
    ctx.fillStyle = COLORS.dim;
    ctx.font = FONT(10);
    const step = domain > 8000 ? (plotW < 420 ? 4000 : 2000) : plotW < 420 ? 2000 : 1000;
    for (let t = 0; t <= domain; t += step) {
      const x = xOf(t);
      const label = t >= 1000 ? `${t / 1000}k` : String(t);
      ctx.fillText(label, Math.min(x - 5, x0 + plotW - ctx.measureText(label).width), bottom + 15);
    }

    if (state.status !== 'ready') {
      ctx.fillStyle = COLORS.dim;
      ctx.font = SANS(14);
      const msg =
        state.status === 'loading'
          ? 'Loading this case’s neuron trace (≈ 140-290 KB)…'
          : state.status === 'missing'
            ? 'No neuron trace was recorded for this case.'
            : `The neuron trace didn’t load: ${state.error}`;
      ctx.fillText(msg, x0 + 14, (top + bottom) / 2);
      return geo;
    }

    const t = state.trace;
    const L = state.layer;
    shadeLoops(ctx, xOf, top, bottom, c.base.loops ?? [], caseExtras(c).extra);
    // the edited run's own (short) loops: a mint strip on the axis
    ctx.fillStyle = COLORS.edited;
    for (const s of c.edited.loops ?? []) ctx.fillRect(xOf(s.start), bottom - 5, Math.max(2, xOf(s.end) - xOf(s.start)), 5);

    // one bucket per 2 CSS px: smooth enough to read, still shows each loop's rhythm
    const BW = 2;
    const buckets = Math.floor(plotW / BW);
    const series = {
      base: bucketStats(t.runs.base.layers[L].push, domain, buckets, { abs: true }),
      counterfactual: bucketStats(t.runs.counterfactual.layers[L].push, domain, buckets, { abs: true }),
      edited: bucketStats(t.runs.edited.layers[L].push, domain, buckets, { abs: true }),
    };
    const ymax = pushScale(state.index, L); // same scale for every case
    const yOf = (v) => bottom - (Math.min(v, ymax) / ymax) * (bottom - top);
    geo.yOf = yOf;

    // y ticks
    ctx.fillStyle = COLORS.dim;
    ctx.font = FONT(10);
    const yStep = ymax > 4 ? 2 : 1;
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 1;
    for (let v = yStep; v < ymax; v += yStep) {
      const y = yOf(v);
      ctx.beginPath();
      ctx.moveTo(x0 + 1, y);
      ctx.lineTo(x0 + plotW, y);
      ctx.stroke();
      ctx.fillText(String(v), x0 - 14, y + 3);
    }
    ctx.fillText('0', x0 - 12, bottom + 3);

    strokeStats(ctx, series.base, x0, yOf, { color: COLORS.base, width: 2.5, step: BW });
    strokeStats(ctx, series.counterfactual, x0, yOf, { color: CF, width: 2.5, dash: [6, 5], step: BW });
    strokeStats(ctx, series.edited, x0, yOf, { color: COLORS.edited, width: 2.5, step: BW });

    // where the edited run stops on its own
    const e = c.edited;
    if (e.finish === 'stop' && e.nTokens < domain) {
      const x = xOf(e.nTokens);
      ctx.strokeStyle = COLORS.editedInk;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, bottom);
      ctx.lineTo(x, bottom - 18);
      ctx.stroke();
      ctx.fillStyle = COLORS.editedInk;
      ctx.font = SANS(11.5);
      const label = '✓ edited stops';
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, Math.min(x + 4, x0 + plotW - tw), bottom - 8);
    }
    return geo;
  }

  function drawChart() {
    if (!chart.canvas || !state.c) return;
    const size = sizeOf(chart);
    const key = `${state.c.id}|${state.status}|${state.layer}|${size.w}x${size.h}@${size.dpr}`;
    if (chart.key !== key) {
      chart.geo = drawChartStatic(size);
      chart.key = key;
    }
    const g = chart.geo;
    paintFrame(chart.canvas, chart.layer, size, {
      x0: g.x0,
      x1: g.x1,
      top: g.top,
      bottom: g.bottom,
      cursorX: g.xOf(state.cursor),
      hoverX: state.hover != null ? g.xOf(state.hover + 0.5) : null,
      dot: true,
    });
  }

  /* ---------- panel: heatmap strip ---------- */
  function rankedChannels() {
    const t = state.trace;
    const L = state.layer;
    return rankChannels(t.runs.base.layers[L].channels.meanPush, t.columns[L]);
  }

  function drawHeatStatic(size) {
    const c = state.c;
    const ctx = layerCtx(heat.layer, size.dpr);
    const { w, h } = size;
    const domain = replayLength(c);
    const { x0, plotW, xOf } = panelX(w, domain);
    const top = 2;
    const bottom = h - 4;
    const geo = { x0, x1: x0 + plotW, xOf, top, bottom };
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 2;
    ctx.strokeRect(x0, top, plotW, bottom - top);
    if (state.status !== 'ready') return geo;

    const t = state.trace;
    const L = state.layer;
    const run = t.runs[state.heatRun];
    const nBins = run.nBins;
    const ranked = rankedChannels();
    // One colour scale for all three runs of this layer (so quieter = paler),
    // saturating at the 98th percentile so a few hot bins don't wash out the rest.
    const all = Float32Array.from(Object.values(t.runs).flatMap((r) => Array.from(r.layers[L].heat))).sort();
    const peak = Math.max(1e-6, all[Math.floor(all.length * 0.98)]);
    const img = new ImageData(nBins, 64);
    ranked.forEach(({ k }, row) => {
      for (let b = 0; b < nBins; b++) {
        const [r, g, bl] = heatRGB(run.layers[L].heat[k * nBins + b] / peak);
        const o = (row * nBins + b) * 4;
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = bl;
        img.data[o + 3] = 255;
      }
    });
    const tile = document.createElement('canvas');
    tile.width = nBins;
    tile.height = 64;
    tile.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    const x1 = xOf(run.nTokens);
    ctx.drawImage(tile, x0, top, x1 - x0, bottom - top);

    // highlight the top channel (row 0 after ranking): sun tab in the gutter + outline
    const rowH = (bottom - top) / 64;
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 - 0.5, top - 0.5, x1 - x0 + 1, rowH + 1);
    ctx.fillStyle = COLORS.sun;
    roundRect(ctx, 2, top - 1, x0 - 6, 17, 8);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = COLORS.ink;
    ctx.font = DISPLAY(12);
    ctx.fillText(`★ #${ranked[0].column}`, 7, top + 12);
    ctx.fillStyle = COLORS.dim;
    ctx.font = FONT(9.5);
    ctx.fillText('64', 2, bottom - 14);
    ctx.fillText('neurons', 2, bottom - 3);
    return geo;
  }

  function drawHeat() {
    if (!heat.canvas || !state.c) return;
    const size = sizeOf(heat);
    const key = `${state.c.id}|${state.status}|${state.layer}|${state.heatRun}|${size.w}x${size.h}@${size.dpr}`;
    if (heat.key !== key) {
      heat.geo = drawHeatStatic(size);
      heat.key = key;
    }
    const g = heat.geo;
    paintFrame(heat.canvas, heat.layer, size, {
      x0: g.x0,
      x1: g.x1,
      top: g.top,
      bottom: g.bottom,
      cursorX: g.xOf(state.cursor),
      hoverX: state.hover != null ? g.xOf(state.hover + 0.5) : null,
    });
  }

  /* ---------- readout ---------- */
  let lastReadout = '';
  function renderReadout() {
    const c = state.c;
    const i = state.hover ?? Math.max(0, Math.floor(state.cursor) - 1);
    const L = state.layer;
    const key = `${c.id}|${state.status}|${L}|${i}|${state.hover != null}|${Math.floor(state.cursor) === 0}`;
    if (key === lastReadout) return;
    lastReadout = key;
    const atStart = state.hover == null && Math.floor(state.cursor) === 0;
    els.tokenHead.textContent = atStart
      ? `token 0 of ${fmtInt(replayLength(c))} · layer ${L} · press play or hover the chart`
      : `token ${fmtInt(i)} · layer ${L} · ${state.hover != null ? 'under the pointer' : 'at the replay cursor'}`;
    for (const run of ['base', 'edited', 'counterfactual']) {
      const row = els.rows[run];
      const tokEl = row.querySelector('.nt-tok');
      const valEl = row.querySelector('.nt-val');
      const tokens = run === 'edited' ? c.edited : c.base;
      if (atStart) {
        tokEl.textContent = '';
        valEl.textContent = '';
        continue;
      }
      if (i >= tokens.nTokens) {
        tokEl.textContent = run === 'edited' && c.edited.finish === 'stop' ? '✓ already stopped' : 'run over';
        valEl.textContent = `after ${fmtInt(tokens.nTokens)} tokens`;
        continue;
      }
      tokEl.textContent = tokenLabel(tokens.tokens[i]);
      if (state.status !== 'ready') {
        valEl.textContent = state.status === 'loading' ? 'loading…' : ' - ';
        continue;
      }
      const s = state.trace.runs[run].layers[L];
      valEl.textContent = `push ${signed(valueAt(s.push, i))} · proj ${signed(valueAt(s.proj, i))} · norm ${valueAt(s.norm, i)?.toFixed(2) ?? ' - '}`;
    }
  }

  function renderStaticLabels() {
    const c = state.c;
    els.caseLabel.textContent = `${c.title.replace(/^MATH-500\s+/, '')} · #${String(c.id).split('-').pop()}${c.kind === 'control' ? ' · control (never loops)' : ''}`;
    if (state.status === 'ready') {
      const top = rankedChannels()[0];
      els.top.textContent = `★ top push neuron in L${state.layer}: #${top.column}  -  ${Math.round(top.share * 100)}% of this base run’s push`;
    } else els.top.textContent = '★ top push neuron:  - ';
    els.sameNote.hidden = state.layer !== 15;
    for (const b of document.querySelectorAll('[data-nt-layer]')) {
      b.setAttribute('aria-pressed', String(Number(b.dataset.ntLayer) === state.layer));
    }
    for (const b of document.querySelectorAll('[data-nt-run]')) {
      b.setAttribute('aria-pressed', String(b.dataset.ntRun === state.heatRun));
    }
  }

  /* ---------- frame ---------- */
  let lastStatic = '';
  function frame() {
    if (!state.c) return;
    drawLane();
    const sk = `${state.c.id}|${state.status}|${state.layer}|${state.heatRun}`;
    if (sk !== lastStatic) {
      lastStatic = sk;
      renderStaticLabels();
      lastReadout = '';
    }
    if (!state.panelVisible) return;
    drawChart();
    drawHeat();
    renderReadout();
    els.scrub.setAttribute('aria-valuemax', String(replayLength(state.c)));
    els.scrub.setAttribute('aria-valuenow', String(Math.floor(state.cursor)));
  }

  /* ---------- input ---------- */
  function wire() {
    for (const b of document.querySelectorAll('[data-nt-layer]')) {
      b.onclick = () => {
        state.layer = Number(b.dataset.ntLayer);
        frame();
      };
    }
    for (const b of document.querySelectorAll('[data-nt-run]')) {
      b.onclick = () => {
        state.heatRun = b.dataset.ntRun;
        frame();
      };
    }
    const scrub = els.scrub;
    let dragging = false;
    const target = (ev) => (ev.target === heat.canvas ? heat.canvas : chart.canvas);
    scrub.addEventListener('pointerdown', (ev) => {
      if (!state.c) return;
      dragging = true;
      scrub.setPointerCapture(ev.pointerId);
      seek(tokenFromX(target(ev), ev.clientX) + 1);
    });
    scrub.addEventListener('pointermove', (ev) => {
      if (!state.c) return;
      const i = tokenFromX(target(ev), ev.clientX);
      if (dragging) {
        state.hover = null;
        seek(i + 1);
      } else if (ev.pointerType === 'mouse') {
        state.hover = i;
        frame();
      }
    });
    const end = () => {
      dragging = false;
    };
    scrub.addEventListener('pointerup', end);
    scrub.addEventListener('pointercancel', end);
    scrub.addEventListener('pointerleave', () => {
      state.hover = null;
      frame();
    });
    scrub.addEventListener('keydown', (ev) => {
      if (!state.c) return;
      const step = ev.shiftKey ? 600 : 60;
      const moves = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step };
      if (ev.key in moves) {
        ev.preventDefault();
        seek(Math.round(state.cursor + moves[ev.key]));
      }
    });
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        state.panelVisible = entries.some((e) => e.isIntersecting);
        if (state.panelVisible) frame();
      });
      io.observe($('neurons'));
    }
  }
  wire();

  return {
    /** Sync to the replay: called on every replay render. */
    update(c, cursor) {
      if (state.c?.id !== c.id) {
        state.c = c;
        state.hover = null;
        state.spike = false;
        load(c);
      }
      state.cursor = cursor;
      frame();
    },
    /**
     * Whether the edited neurons are firing hard in the base run right now
     * (smoothed layer-19 |push| with hysteresis, so it never flickers).
     */
    spikeAt(cursor) {
      const smooth = state.status === 'ready' ? perCase.get(state.c.id)?.smooth : null;
      if (!smooth) return (state.spike = false);
      const k = Math.min(Math.floor(cursor), smooth.length);
      state.spike = k > 0 && hysteresis(state.spike, smooth[k - 1], SPIKE);
      return state.spike;
    },
    resize() {
      state.sizeGen++;
      frame();
    },
  };
}

/** The replay never loaded: say so in the panel instead of "loading…". */
export function renderNeuronLoadError() {
  const badge = $('nt-badge');
  badge.hidden = false;
  badge.textContent = 'needs the recording';
  badge.className = 'badge recording';
  $('nt-token').textContent = 'The neuron trace indexes into the recorded replay, which didn’t load.';
}
