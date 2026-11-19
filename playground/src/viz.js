/**
 * viz.js  -  canvas/DOM visualizations for the showcase.
 *
 * Everything here is presentation-only: it draws the live inference traces
 * (sparkline, loop meter, divergence timeline) and the precomputed weight-space
 * weight-diff panel. The neuron-trace panel lives in neuron-view.js. No engine access.
 */

import { niceStep, rank1Summary } from './lib.js';

// Playful light palette  -  keep in sync with styles.css custom properties.
export const COLORS = {
  base: '#ff6f59',
  edited: '#1fb589',
  editedInk: '#138a67', // edited colour dark enough for text on paper
  delta: '#8c6bff', // the edit itself (weight delta), distinct from either model
  ink: '#2b2340',
  dim: '#7d7290',
  dim2: '#a297b3',
  line: '#eadbc2',
  track: '#f3e5cc',
  think: '#d9cdb9',
  text: '#2b2340',
  bad: '#e5484d',
  sun: '#ffd23f',
};
export const FONT = (px) => `600 ${px}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`;
export const SANS = (px) => `700 ${px}px Nunito, ui-rounded, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
export const DISPLAY = (px) => `600 ${px}px Fredoka, Nunito, ui-rounded, system-ui, sans-serif`;

/** Size a canvas to its CSS box at device-pixel resolution; returns a context. */
export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function accentFor(which) {
  return which === 'edited' ? COLORS.edited : COLORS.base;
}

/** Per-chunk latency sparkline (ms). */
export function drawSparkline(canvas, values, which) {
  const { ctx, w, h } = setupCanvas(canvas);
  const color = accentFor(which);
  if (!values.length) {
    ctx.strokeStyle = COLORS.line;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h - 1);
    ctx.lineTo(w, h - 1);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.dim2;
    ctx.font = SANS(11);
    ctx.fillText('time per chunk (ms) appears here during a run', 2, h / 2 + 2);
    return;
  }
  const max = Math.max(1, ...values);
  const n = values.length;
  const xOf = (i) => (n <= 1 ? 0 : (i / (n - 1)) * (w - 2)) + 1;
  // baseline
  ctx.strokeStyle = COLORS.line;
  ctx.beginPath();
  ctx.moveTo(0, h - 1);
  ctx.lineTo(w, h - 1);
  ctx.stroke();
  // area
  ctx.beginPath();
  ctx.moveTo(xOf(0), h);
  for (let i = 0; i < n; i++) ctx.lineTo(xOf(i), h - (values[i] / max) * (h - 4));
  ctx.lineTo(xOf(n - 1), h);
  ctx.closePath();
  ctx.fillStyle = color + '40';
  ctx.fill();
  // line
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xOf(i);
    const y = h - (values[i] / max) * (h - 4);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.25;
  ctx.lineJoin = 'round';
  ctx.stroke();
  ctx.fillStyle = COLORS.dim;
  ctx.font = FONT(10);
  ctx.fillText(`ms/chunk · max ${max.toFixed(0)}`, 2, 10);
}

/** Repetition/loop meter 0..1. */
export function drawLoopMeter(canvas, score, which) {
  const { ctx, w, h } = setupCanvas(canvas);
  const s = Math.max(0, Math.min(1, score || 0));
  // track (the CSS border draws the ink outline)
  ctx.fillStyle = COLORS.track;
  roundRect(ctx, 0, 0, w, h, h / 2);
  ctx.fill();
  // fill (green -> coral -> red as it climbs)
  const color = s < 0.25 ? COLORS.edited : s < 0.55 ? COLORS.base : COLORS.bad;
  ctx.fillStyle = color;
  roundRect(ctx, 0, 0, Math.max(0, w * s), h, h / 2);
  ctx.fill();
}

export function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Horizontal geometry of the timeline plot (shared with the replay scrubber). */
export function timelineGeometry(width) {
  const padL = 70, padR = 16;
  return { padL, padR, plotW: Math.max(1, width - padL - padR) };
}

/**
 * Divergence timeline: one lane per model, x = token (or stream chunk) index,
 * grey = thinking, colour = answer. Shows base running to the token cap while
 * edited stops. With no traces it draws an explicit empty state.
 *
 * A trace is { n, thinkEnd, finish, stopped?, loops? }. `opts` (replay only):
 * maxN (x-axis extent), cap (token cap mark), cursor (playhead; the unplayed
 * part is faded). Recorded loop segments are drawn as coral hatching.
 */
export function drawTimeline(canvas, base, edited, opts = {}) {
  const { ctx, w, h } = setupCanvas(canvas);
  const hasData = Boolean(base?.n || edited?.n);
  const replay = opts.cursor != null;
  const maxN = Math.max(1, opts.maxN ?? 0, base?.n || 0, edited?.n || 0);
  const { padL, padR, plotW } = timelineGeometry(w);
  const padT = replay ? 22 : 8, padB = 24;
  const xOf = (i) => padL + (i / maxN) * plotW;
  const laneH = 26;
  const lanes = [
    { label: 'base', trace: base, y: padT + 2, color: COLORS.base, ink: COLORS.base },
    { label: 'edited', trace: edited, y: padT + 2 + laneH + 14, color: COLORS.edited, ink: COLORS.editedInk },
  ];
  const laneLabel = (lane) => {
    ctx.fillStyle = lane.ink;
    ctx.font = DISPLAY(15);
    ctx.fillText(lane.label, 6, lane.y + laneH / 2 + 5);
  };

  if (!hasData) {
    for (const lane of lanes) {
      laneLabel(lane);
      ctx.strokeStyle = COLORS.dim2;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      roundRect(ctx, padL + 1, lane.y + 1, plotW - 2, laneH - 2, laneH / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = COLORS.dim;
    ctx.font = SANS(13);
    let msg = 'Run the comparison to draw both output streams here.';
    if (ctx.measureText(msg).width > plotW) msg = 'Run the comparison to see this.';
    const tw = ctx.measureText(msg).width;
    ctx.fillText(msg, Math.max(padL, padL + (plotW - tw) / 2), h - 6);
    return;
  }

  // x axis
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(padL, h - padB);
  ctx.lineTo(w - padR, h - padB);
  ctx.stroke();
  ctx.fillStyle = COLORS.dim;
  ctx.font = FONT(10);
  const step = niceStep(maxN) * (plotW < 420 && maxN > 1000 ? 2 : 1);
  for (let t = 0; t <= maxN; t += step) {
    const x = xOf(t);
    const label = replay && t >= 1000 ? `${t / 1000}k` : String(t);
    ctx.fillText(label, x - 5, h - padB + 15);
    ctx.beginPath();
    ctx.moveTo(x, h - padB);
    ctx.lineTo(x, h - padB + 4);
    ctx.stroke();
  }

  for (const lane of lanes) {
    laneLabel(lane);
    const t = lane.trace;
    if (!t || !t.n) {
      ctx.fillStyle = COLORS.dim;
      ctx.font = SANS(13);
      ctx.fillText('not run', padL + 6, lane.y + laneH / 2 + 5);
      continue;
    }
    const thinkEnd = t.thinkEnd == null ? t.n : t.thinkEnd;
    const x0 = xOf(0);
    const endX = xOf(t.n);
    const barW = Math.max(laneH, endX - x0);
    // clip to a chunky pill, then paint thinking + answer inside it
    ctx.save();
    roundRect(ctx, x0, lane.y, barW, laneH, laneH / 2);
    ctx.clip();
    ctx.fillStyle = COLORS.think;
    ctx.fillRect(x0, lane.y, barW, laneH);
    if (thinkEnd < t.n) {
      ctx.fillStyle = lane.color;
      ctx.fillRect(xOf(thinkEnd), lane.y, Math.max(1, endX - xOf(thinkEnd)), laneH);
    }
    for (const seg of t.loops ?? []) hatch(ctx, xOf(seg.start), lane.y, Math.max(2, xOf(seg.end) - xOf(seg.start)), laneH);
    ctx.restore();
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 2.5;
    roundRect(ctx, x0, lane.y, barW, laneH, laneH / 2);
    ctx.stroke();
    if (replay && t.thinkEnd != null) {
      // "</think>": where the model stops thinking and starts its answer
      const x = xOf(t.thinkEnd);
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, lane.y - 4);
      ctx.lineTo(x, lane.y + laneH + 4);
      ctx.stroke();
    }
    const verdict =
      t.finish === 'length' ? '⚠ token cap' : t.finish === 'stop' ? '✓ stopped' : t.stopped ? '■ stopped by you' : '';
    if (verdict) {
      ctx.font = SANS(13);
      const tw = ctx.measureText(verdict).width + 16;
      const vx = Math.max(x0, Math.min(x0 + barW + 8, w - padR - tw));
      const inside = vx < x0 + barW;
      const fill = t.finish === 'length' ? COLORS.bad : t.finish === 'stop' ? COLORS.edited : '#ffffff';
      ctx.fillStyle = fill;
      roundRect(ctx, vx, lane.y + 2, tw, laneH - 4, (laneH - 4) / 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = inside ? 2.5 : 2;
      ctx.stroke();
      ctx.fillStyle = fill === '#ffffff' ? COLORS.ink : '#ffffff';
      ctx.fillText(verdict, vx + 8, lane.y + laneH / 2 + 5);
    }
  }

  if (!replay) return;

  // top marker row: thinkEnd ticks + the token cap
  ctx.font = FONT(10);
  if (opts.cap != null) {
    const x = xOf(opts.cap);
    ctx.strokeStyle = COLORS.bad;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, padT - 4);
    ctx.lineTo(x, h - padB);
    ctx.stroke();
    ctx.setLineDash([]);
    const label = `cap ${opts.cap.toLocaleString('en-US')}`;
    ctx.fillStyle = COLORS.bad;
    ctx.fillText(label, Math.min(x, w - 2) - ctx.measureText(label).width - 4, 11);
  }
  const thinkLabels = lanes.filter((l) => l.trace?.thinkEnd != null);
  if (thinkLabels.length) {
    ctx.fillStyle = COLORS.dim;
    const x = xOf(Math.min(...thinkLabels.map((l) => l.trace.thinkEnd)));
    ctx.fillText('</think> ↓', Math.max(padL, x - 24), 11);
  }

  // playhead: fade what has not been played yet, then draw the cursor
  const cx = xOf(Math.max(0, Math.min(maxN, opts.cursor)));
  ctx.fillStyle = 'rgba(255, 250, 241, 0.62)';
  ctx.fillRect(cx, padT - 2, w - padR - cx + padR, h - padB - padT + 2);
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, padT - 2);
  ctx.lineTo(cx, h - padB);
  ctx.stroke();
  ctx.fillStyle = COLORS.sun;
  ctx.beginPath();
  ctx.arc(cx, h - padB, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.stroke();
}

/** Coral diagonal hatching marking a recorded verbatim-loop segment. */
function hatch(ctx, x, y, w, h) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.fillStyle = 'rgba(255, 111, 89, 0.55)';
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = COLORS.base;
  ctx.lineWidth = 3;
  for (let i = x - h; i < x + w + h; i += 9) {
    ctx.beginPath();
    ctx.moveTo(i, y + h);
    ctx.lineTo(i + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

/* ---------- precomputed weight diff ---------- */

/**
 * Render the precomputed weight diff. Canvases are attached before drawing so
 * they measure their real CSS size. Returns a redraw function for resizes.
 */
export function renderWeightDiff(root, data, badgeEl, noteEl) {
  setBadge(badgeEl, 'precomputed', 'pre');
  if (noteEl) {
    noteEl.textContent =
      `${data.quant ?? ''} · ${data.tensors.length} tensors · exact base↔edited delta`.replace(/^ · /, '');
  }
  root.innerHTML = '';
  const r1 = rank1Summary(data);
  if (r1) {
    // The headline fact of the bf16 diff: every changed column is the same direction.
    const box = document.createElement('div');
    box.className = 'rank1';
    const eq = document.createElement('span');
    eq.className = 'rank1-eq';
    eq.textContent = 'w′ = w − α (u·w) u';
    const line = document.createElement('p');
    const neurons = r1.columnsPerTensor != null ? `${r1.columnsPerTensor} neurons` : 'the edited neurons';
    const b = document.createElement('b');
    b.textContent = 'Rank 1: ';
    line.append(
      b,
      `the whole change is one direction, scaled by ~${Math.round(r1.alpha)}, written into ${neurons}` +
        (r1.tensors > 1 ? ` (in each of ${r1.tensors} tensors).` : '.'),
    );
    const sub = document.createElement('p');
    sub.className = 'rank1-sub';
    sub.textContent =
      `u is one 1,024-d direction in the residual stream; each edited neuron's output weights w lose ` +
      `α ≈ ${r1.alpha.toFixed(2)} times their component along u. Rank-1 share of the change: ` +
      `≥ ${(r1.minRank1Energy * 100).toFixed(3)}%.`;
    const text = document.createElement('div');
    text.append(line, sub);
    box.append(eq, text);
    root.appendChild(box);
  }
  const grid = document.createElement('div');
  grid.className = 'panel-grid';
  const draws = [];
  for (const tensor of data.tensors) {
    const card = document.createElement('div');
    card.className = 'tensor';
    const pct = tensor.pctChanged != null ? tensor.pctChanged.toFixed(2) : '?';
    const h3 = document.createElement('h3');
    h3.textContent = tensor.name;
    const stats = document.createElement('div');
    stats.className = 'stats';
    stats.textContent =
      `${fmtInt(tensor.entriesChanged)} / ${fmtInt(tensor.totalEntries)} weights (${pct}%) · ` +
      `${tensor.changedColumns.length} neurons · max|Δ| ${tensor.maxAbsDelta?.toFixed(3) ?? '?'}`;
    const ruler = document.createElement('canvas');
    ruler.className = 'ruler';
    const bars = document.createElement('canvas');
    bars.className = 'bars';
    card.append(h3, stats, ruler, bars);
    grid.appendChild(card);
    draws.push(() => {
      drawColumnRuler(ruler, tensor);
      drawDeltaBars(bars, tensor);
    });
  }
  root.appendChild(grid);
  const redraw = () => draws.forEach((d) => d());
  redraw();
  return redraw;
}

function drawColumnRuler(canvas, tensor) {
  const { ctx, w, h } = setupCanvas(canvas);
  const inDim = tensor.inputDim ?? Math.max(...(tensor.shape ?? [tensor.changedColumns.length]));
  ctx.fillStyle = COLORS.track;
  roundRect(ctx, 0, 2, w, h - 12, 6);
  ctx.fill();
  ctx.fillStyle = COLORS.delta;
  for (const col of tensor.changedColumns) {
    const x = (col / inDim) * w;
    ctx.fillRect(x, 2, Math.max(1.5, w / inDim), h - 12);
  }
  ctx.fillStyle = COLORS.dim;
  ctx.font = FONT(10);
  ctx.fillText('neuron 0', 1, h - 1);
  const right = String(inDim - 1);
  ctx.fillText(right, w - ctx.measureText(right).width - 1, h - 1);
}

function drawDeltaBars(canvas, tensor) {
  const { ctx, w, h } = setupCanvas(canvas);
  const vals = tensor.deltaNorm || [];
  if (!vals.length) return;
  const max = Math.max(1e-9, ...vals);
  const top = 14;
  const bw = w / vals.length;
  for (let i = 0; i < vals.length; i++) {
    const bh = (vals[i] / max) * (h - top);
    ctx.fillStyle = COLORS.delta;
    roundRect(ctx, i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh, Math.min(3, bw / 2));
    ctx.fill();
  }
  ctx.fillStyle = COLORS.dim;
  ctx.font = FONT(10);
  ctx.fillText('|Δ| per edited neuron', 2, 10);
}

/* ---------- "data being recorded" empty state ---------- */

const PENDING = {
  weightDiff: {
    title: 'Weight diff is being computed offline',
    what: 'Here you will see the 64 neurons (of 3,584) whose output weights were changed in each of layers 15 and 19, and the size of each change.',
    why: 'It is computed from the original full-precision checkpoints, because the two in-browser 4-bit files are quantized differently and would make every weight look changed.',
    rows: 2,
  },
};

/** Clearly non-data placeholder: hatched skeleton + what will appear and why. */
export function renderPending(root, kind, badgeEl, noteEl) {
  const p = PENDING[kind];
  setBadge(badgeEl, 'data being recorded', 'recording');
  if (noteEl) noteEl.textContent = '';
  root.innerHTML = '';
  const box = document.createElement('div');
  box.className = 'pending';
  const sk = document.createElement('div');
  sk.className = 'pending-skeleton' + (p.rows > 1 ? ' two' : '');
  for (let i = 0; i < p.rows; i++) sk.appendChild(document.createElement('div'));
  const text = document.createElement('div');
  text.className = 'pending-text';
  const h4 = document.createElement('h4');
  h4.textContent = p.title;
  const what = document.createElement('p');
  what.textContent = p.what;
  const why = document.createElement('p');
  why.className = 'why';
  why.textContent = p.why;
  text.append(h4, what, why);
  box.append(sk, text);
  root.appendChild(box);
}

function setBadge(el, text, kind) {
  if (!el) return;
  el.textContent = text;
  el.className = 'badge ' + kind;
}

/* ---------- data loading ---------- */

async function fetchJson(path) {
  try {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/**
 * Precomputed weight-diff data; a missing file is `null` (rendered as "being
 * computed"). The neuron trace has its own lazy loader (neuron-trace.js).
 */
export async function loadShowcaseData() {
  const weightDiff = await fetchJson('data/weight-diff.json');
  return {
    weightDiff: weightDiff && !weightDiff.placeholder && weightDiff.tensors?.length ? weightDiff : null,
  };
}

function fmtInt(n) {
  if (n == null) return '?';
  return n.toLocaleString('en-US');
}
