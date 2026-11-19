/**
 * main.js  -  Weight-Space Playground (Vite entry).
 *
 * The page opens on a recorded replay of both models (replay-view.js, lazily
 * fetched data/replay.json); the live run below is opt-in.
 *
 * Streams base vs edited Qwen3.5-0.8B in-browser (wllama WebGPU) and visualizes
 * the inference process: thinking/answer split, latency sparkline, repetition
 * ("loop") meter, and a divergence timeline. The weight-diff panel is driven
 * by precomputed JSON (public/data/weight-diff.json) via viz.js; the neuron
 * trace panel is synced to the replay cursor (neuron-view.js, mounted by
 * replay-view.js). Pure helpers live in lib.js.
 *
 * wllama is loaded at runtime from jsdelivr as an external ES module (kept out
 * of the bundle with @vite-ignore) so the engine version is pinned by URL, not
 * by our bundle.
 */
import './styles.css';
import {
  drawSparkline,
  drawLoopMeter,
  drawTimeline,
  renderWeightDiff,
  renderPending,
  loadShowcaseData,
} from './viz.js';
import {
  clampInt,
  clampFloat,
  fmtBytes,
  shortName,
  isNearBottom,
  createLoopMeter,
} from './lib.js';
import { mountReplay } from './replay-view.js';
import { initAnalytics } from './analytics.js';

const WLLAMA_VERSION = '3.13.0';
const WLLAMA_BASE = `https://cdn.jsdelivr.net/npm/@wllama/wllama@${WLLAMA_VERSION}/esm`;

const DEFAULT_PROMPT =
  'What is $2^{10} + 24$? Please reason step by step, and put your final answer within \\boxed{}.';

// Pinned HF revisions: immutable URLs that are also the wllama OPFS cache keys.
// Do NOT change them between app versions or every visitor re-downloads ~1 GB.
const MODELS = {
  base: {
    label: 'Base',
    src: 'unsloth/Qwen3.5-0.8B-GGUF',
    url: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf',
  },
  edited: {
    label: 'Weight-edited',
    src: 'NotCherry/…-GGUF',
    url: 'https://huggingface.co/NotCherry/Qwen3.5-0.8B-weight-space-programming-GGUF/resolve/aded4c8fa22992e397cdad51a42299abf463ba52/Qwen3.5-0.8B-Weight-Space-Programming-Q4_K_M.gguf',
  },
};

// Edited layers (of 24) for the hero scale visual.
const N_LAYERS = 24;
const EDITED_LAYERS = [15, 19];

const $ = (id) => document.getElementById(id);
const WHICHES = ['base', 'edited'];

const msg = (e) => (e instanceof Error ? e.message : String(e));

/* ---------- state ---------- */
let wllama = null; // the externally loaded wllama ES module
let modelManager = null;
const instances = {};
const panes = {};
const traces = {};
let abortCtl = null;
let panelRedraws = []; // weight-space panel redraws (resize)
let replayRedraw = () => {}; // recorded replay stage redraw (resize)

/** Load the pinned wllama ES module from jsdelivr, left out of the bundle. */
async function loadEngine() {
  const [mod, wasm] = await Promise.all([
    import(/* @vite-ignore */ `${WLLAMA_BASE}/index.js`),
    import(/* @vite-ignore */ `${WLLAMA_BASE}/wasm-from-cdn.js`),
  ]);
  return { ...mod, WasmFromCDN: wasm.default };
}

function initPanes() {
  for (const which of WHICHES) {
    panes[which] = {
      meta: $(`meta-${which}`),
      thinkEl: $(`think-${which}`),
      answerEl: $(`answer-${which}`),
      scroll: $(`scroll-${which}`),
      spark: $(`spark-${which}`),
      loop: $(`loop-${which}`),
      loopLabel: $(`looplabel-${which}`),
      badge: $(`finish-${which}`),
    };
  }
}

/**
 * Pin the live pane head and metrics line to their tallest content. The finish
 * badge ("not loaded" → "⚠ hit token limit") and the metrics string ("not
 * loaded" → "16,384 chunks · 16,384 tokens · 1,920.0 tok/s · 12.3s") both gain
 * text during a run; without the reservation they wrap and grow the cards,
 * shifting everything below them.
 */
const LIVE_BADGES = ['✓ stopped on its own', '⚠ hit token limit', 'not loaded'];
const LIVE_METRICS = ['not loaded', '16,384 chunks · 16,384 tokens · 1,920.0 tok/s · 12.3s'];

function reserveLiveHeights() {
  for (const which of WHICHES) {
    const { badge, meta } = panes[which];
    const head = badge.closest('.pane-head');
    head.style.minHeight = '';
    const badgeText = badge.textContent;
    const badgeCls = badge.className;
    let h = 0;
    for (const label of LIVE_BADGES) {
      badge.textContent = label;
      h = Math.max(h, head.offsetHeight);
    }
    badge.textContent = badgeText;
    badge.className = badgeCls;
    head.style.minHeight = `${h}px`;

    meta.style.minHeight = '';
    const metaText = meta.textContent;
    let mh = 0;
    for (const label of LIVE_METRICS) {
      meta.textContent = label;
      mh = Math.max(mh, meta.offsetHeight);
    }
    meta.textContent = metaText;
    meta.style.minHeight = `${mh}px`;
  }
}

function newTrace(which) {
  return {
    which,
    text: '',
    think: '',
    answer: '',
    n: 0,
    thinkEnd: null,
    events: [],
    dts: [],
    start: 0,
    end: 0,
    finish: null,
    stopped: false,
    tokens: 0,
    tps: null,
    loop: 0,
    loopMeter: createLoopMeter({ n: 4, window: 80 }),
  };
}

function setStatus(text, kind) {
  const el = $('status');
  el.className = 'status' + (kind ? ' ' + kind : '');
  el.textContent = text;
}

function reportError(e) {
  if (isAbort(e)) {
    setStatus('Stopped.', null);
    return;
  }
  console.error(e);
  setStatus('Error: ' + msg(e), 'bad');
}

/* ---------- engine ---------- */
async function ensureModel(which) {
  if (instances[which]) return instances[which];
  const pane = panes[which];
  const w = new wllama.Wllama(wllama.WasmFromCDN, {
    modelManager,
    logger: wllama.LoggerWithoutDebug,
  });
  const gpu = w.isSupportWebGPU();
  if (!gpu) setStatus('WebGPU not available  -  running on CPU (very slow).', 'bad');
  try {
    await w.loadModelFromUrl(MODELS[which].url, {
      n_ctx: clampInt($('nctx').value, 512, 32768, 8192),
      n_gpu_layers: gpu ? clampInt($('ngpu').value, 0, 100000, 9999) : 0,
      useCache: true,
      progressCallback: ({ loaded, total }) => {
        const pct = total ? (loaded / total) * 100 : 0;
        pane.meta.textContent = `download ${pct.toFixed(0)}% · ${fmtBytes(loaded)}/${fmtBytes(total)}`;
      },
    });
  } catch (err) {
    try { await w.exit(); } catch (_) {}
    throw err;
  }
  instances[which] = w;
  pane.meta.textContent = 'loaded';
  return w;
}

async function unload(which) {
  const w = instances[which];
  if (!w) return;
  try { await w.exit(); } catch (_) {}
  delete instances[which];
}

/* ---------- runs ---------- */
function readOptions() {
  return {
    prompt: $('prompt').value.trim(),
    maxTokens: clampInt($('maxnt').value, 16, 65536, 4096),
    temperature: clampFloat($('temp').value, 0, 2, 0.6),
    seed: clampInt($('seed').value, 0, 1e9, 0),
  };
}

async function runPane(which, opts) {
  const pane = panes[which];
  const trace = newTrace(which);
  traces[which] = trace;

  pane.thinkEl.textContent = '';
  pane.thinkEl.hidden = true;
  pane.answerEl.textContent = '';
  pane.answerEl.classList.add('muted');
  pane.badge.textContent = 'loading…';
  pane.badge.className = 'finish';
  drawTimeline($('timeline'), traces.base, traces.edited);

  try {
    await generate(which, trace, pane, opts);
  } catch (e) {
    trace.end = performance.now();
    if (isAbort(e) || opts.signal.aborted) {
      trace.stopped = true;
      finalizePane(which, trace);
    } else {
      pane.badge.textContent = '✕ error';
      pane.badge.className = 'finish bad';
      pane.meta.textContent = 'error: ' + msg(e);
    }
    throw e;
  }
  trace.end = performance.now();
  if (opts.signal.aborted && !trace.finish) trace.stopped = true;
  finalizePane(which, trace);
  return trace;
}

const isAbort = (e) => e && e.name === 'AbortError';

async function generate(which, trace, pane, opts) {
  const w = await ensureModel(which);
  pane.badge.textContent = 'generating…';
  trace.start = performance.now();
  let last = trace.start;

  await w.createChatCompletion({
    messages: [{ role: 'user', content: opts.prompt }],
    chat_template_kwargs: { enable_thinking: true },
    max_tokens: opts.maxTokens,
    temperature: opts.temperature,
    top_p: 0.95,
    top_k: 20,
    seed: opts.seed,
    stream: true,
    abortSignal: opts.signal,
    onData: (chunk) => {
      const choice = chunk.choices && chunk.choices[0];
      const delta = (choice && choice.delta) || {};
      if (delta.reasoning_content) appendChunk(trace, pane, 'think', delta.reasoning_content);
      if (delta.content) appendChunk(trace, pane, 'answer', delta.content);
      if (choice && choice.finish_reason) trace.finish = choice.finish_reason;
      if (chunk.usage && typeof chunk.usage.completion_tokens === 'number') {
        trace.tokens = chunk.usage.completion_tokens;
      }
      const t = chunk.timings || {};
      if (t.predicted_per_second) trace.tps = t.predicted_per_second;
      else if (t.predicted_per_sec) trace.tps = t.predicted_per_sec;

      const now = performance.now();
      trace.dts.push(now - last);
      last = now;
      trace.n++;
      updatePaneLive(which, trace, now);
    },
  });
}

function appendChunk(trace, pane, kind, text) {
  // Follow the stream only if the reader is already at the bottom.
  const follow = isNearBottom(pane.scroll);
  trace.loop = trace.loopMeter.push(text);
  if (kind === 'think') {
    trace.think += text;
    trace.text += text;
    if (pane.thinkEl.hidden) pane.thinkEl.hidden = false;
    pane.thinkEl.appendChild(document.createTextNode(text));
  } else {
    if (trace.thinkEnd === null) trace.thinkEnd = trace.n;
    trace.answer += text;
    trace.text += text;
    pane.answerEl.classList.remove('muted');
    pane.answerEl.appendChild(document.createTextNode(text));
  }
  if (follow) pane.scroll.scrollTop = pane.scroll.scrollHeight;
  trace.events.push({ i: trace.n, kind });
}

let timelineQueued = false;
function queueTimeline() {
  if (timelineQueued) return;
  timelineQueued = true;
  requestAnimationFrame(() => {
    timelineQueued = false;
    drawTimeline($('timeline'), traces.base, traces.edited);
  });
}

function updatePaneLive(which, trace, now) {
  const pane = panes[which];
  const sec = (now - trace.start) / 1000;
  const tps = trace.tps ?? (trace.n && sec > 0 ? trace.n / sec : null);
  pane.meta.textContent =
    `${trace.n} chunks · ${sec.toFixed(1)}s` + (tps ? ` · ${tps.toFixed(1)} tok/s` : '');
  drawSparkline(pane.spark, trace.dts, which);
  drawLoopMeter(pane.loop, trace.loop, which);
  pane.loopLabel.textContent = `${Math.round(trace.loop * 100)}%`;
  queueTimeline();
}

function finalizePane(which, trace) {
  const pane = panes[which];
  const sec = (trace.end - trace.start) / 1000;
  const tps = trace.tps ?? (trace.n && sec > 0 ? trace.n / sec : null);
  trace.loop = trace.loopMeter.flush();
  const fr = trace.finish;
  pane.badge.textContent =
    fr === 'length' ? '⚠ hit token limit' : fr === 'stop' ? '✓ stopped on its own' : trace.stopped ? '■ stopped by you' : 'done';
  pane.badge.className = 'finish ' + (fr === 'length' ? 'bad' : fr === 'stop' ? 'good' : '');
  pane.meta.textContent =
    `${trace.n} chunks` +
    (trace.tokens ? ` · ${trace.tokens} tokens` : '') +
    (tps ? ` · ${tps.toFixed(1)} tok/s` : '') +
    ` · ${sec.toFixed(1)}s`;
  drawSparkline(pane.spark, trace.dts, which);
  drawLoopMeter(pane.loop, trace.loop, which);
  pane.loopLabel.textContent = `${Math.round(trace.loop * 100)}%`;
  drawTimeline($('timeline'), traces.base, traces.edited);
}

function setBusy(busy) {
  $('compare').disabled = busy;
  $('stop').disabled = !busy;
  document.querySelectorAll('[data-run]').forEach((b) => (b.disabled = busy));
}

async function compare() {
  const o = readOptions();
  if (!o.prompt) return;
  abortCtl = new AbortController();
  setBusy(true);
  await requestPersistentStorage();
  setStatus('Running base, then edited (one at a time to limit GPU memory)…');
  try {
    for (const which of WHICHES) {
      if (abortCtl.signal.aborted) break;
      await runPane(which, { ...o, signal: abortCtl.signal });
      if (!$('keep').checked) await unload(which);
    }
    if (abortCtl.signal.aborted) setStatus('Stopped.', null);
    else setStatus('Done. Compare how each run ended: ⚠ token cap vs ✓ stopped on its own.', 'good');
  } catch (e) {
    reportError(e);
  } finally {
    setBusy(false);
    refreshCache();
  }
}

async function runSingle(which) {
  const o = readOptions();
  if (!o.prompt) return;
  abortCtl = new AbortController();
  setBusy(true);
  await requestPersistentStorage();
  setStatus(`Running ${which}…`);
  try {
    await runPane(which, { ...o, signal: abortCtl.signal });
    if (!$('keep').checked) await unload(which);
    setStatus(abortCtl.signal.aborted ? 'Stopped.' : 'Done.', abortCtl.signal.aborted ? null : 'good');
  } catch (e) {
    reportError(e);
  } finally {
    setBusy(false);
    refreshCache();
  }
}

/** Ask once, on the first user-started run, to keep the ~1 GB cache from eviction. */
let persistRequested = false;
async function requestPersistentStorage() {
  if (persistRequested) return;
  persistRequested = true;
  try {
    if (navigator.storage && navigator.storage.persist) await navigator.storage.persist();
  } catch (_) {}
}

/* ---------- cache ---------- */
async function refreshCache() {
  if (!modelManager) return;
  try {
    const models = await modelManager.getModels({ includeInvalid: true });
    if (!models.length) {
      $('cachelist').textContent = 'No models cached yet.';
      return;
    }
    let total = 0;
    const parts = models.map((m) => {
      const s = Math.max(0, m.size || 0);
      total += s;
      return shortName(m.url);
    });
    $('cachelist').textContent = `cached: ${parts.join(', ')}  -  ${fmtBytes(total)}`;
  } catch (_) {
    $('cachelist').textContent = '';
  }
}

async function clearCache() {
  if (!modelManager) return;
  if (instances.base || instances.edited) {
    if (!confirm('Unload models and delete cached downloads?')) return;
  } else if (!confirm('Delete cached model downloads from this browser?')) return;
  await unload('base');
  await unload('edited');
  await modelManager.clear();
  refreshCache();
}

/* ---------- static visuals ---------- */
function renderLayerScale() {
  const root = $('layers');
  if (!root) return;
  for (let i = 0; i < N_LAYERS; i++) {
    const col = document.createElement('div');
    col.className = 'layer' + (EDITED_LAYERS.includes(i) ? ' edited' : '');
    col.title = EDITED_LAYERS.includes(i)
      ? `layer ${i}: 64 of 3,584 ffn_down input neurons edited`
      : `layer ${i}: untouched`;
    if (EDITED_LAYERS.includes(i)) {
      const tag = document.createElement('span');
      tag.textContent = String(i);
      col.appendChild(tag);
    }
    root.appendChild(col);
  }
}

function drawEmptyLive() {
  for (const which of WHICHES) {
    const t = traces[which];
    drawSparkline(panes[which].spark, t ? t.dts : [], which);
    drawLoopMeter(panes[which].loop, t ? t.loop : 0, which);
  }
  drawTimeline($('timeline'), traces.base, traces.edited);
}

let resizeQueued = false;
function onResize() {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => {
    resizeQueued = false;
    drawEmptyLive();
    reserveLiveHeights();
    replayRedraw();
    panelRedraws.forEach((r) => r());
  });
}

/* ---------- wiring ---------- */
function wire() {
  $('compare').onclick = compare;
  $('stop').onclick = () => abortCtl?.abort();
  $('clear').onclick = clearCache;
  document.querySelectorAll('[data-run]').forEach((b) => {
    b.onclick = () => runSingle(b.getAttribute('data-run'));
  });
  $('prompt').value = DEFAULT_PROMPT;
  window.addEventListener('resize', onResize);
}

(async function init() {
  initAnalytics();
  initPanes();
  wire();
  renderLayerScale();
  drawEmptyLive();
  reserveLiveHeights();
  document.fonts?.ready.then(reserveLiveHeights); // wrapping depends on the font
  replayRedraw = mountReplay();

  // Weight-space panels first: they need no engine.
  loadShowcaseData().then((d) => {
    panelRedraws = [];
    if (d.weightDiff) panelRedraws.push(renderWeightDiff($('wd-body'), d.weightDiff, $('wd-badge'), $('wd-note')));
    else renderPending($('wd-body'), 'weightDiff', $('wd-badge'), $('wd-note'));
  });

  try {
    wllama = await loadEngine();
    modelManager = new wllama.ModelManager({ cacheManager: new wllama.CacheManager() });
  } catch (err) {
    setStatus('Failed to load the wllama engine: ' + msg(err), 'bad');
    return;
  }
  refreshCache();

  if (!('gpu' in navigator)) {
    setStatus('WebGPU not available  -  the demo still runs on CPU WASM, but very slowly.', 'bad');
  } else {
    setStatus('Engine ready. The first live run downloads ~1.05 GB once; later visits load from cache.', null);
  }
})();
