/**
 * replay-view.js  -  DOM wiring of the recorded replay stage (#replay) and the
 * honest data card (#honest). All decisions about "what is true at token k"
 * live in replay.js (pure, unit-tested); this file only renders them.
 *
 * replay.json (~1.1 MB) is fetched lazily after first paint, with a playful
 * loading state. The canvas renderers (timeline, loop meter) are the same
 * ones the live run uses (viz.js). The neuron trace (neuron-view.js) follows
 * the same cursor: a lane under the timeline and the #neurons panel.
 */
import { drawTimeline, drawLoopMeter, timelineGeometry } from './viz.js';
import { isNearBottom } from './lib.js';
import { createNeuronView, renderNeuronLoadError } from './neuron-view.js';
import {
  SPEEDS,
  DEFAULT_SPEED,
  TOKENS_PER_SECOND_1X,
  advanceCursor,
  clampCursor,
  replayLength,
  capOf,
  prepareCase,
  textOf,
  repetitionAt,
  textSpans,
  countersAt,
  runVerdict,
  loopPhrase,
  parseReplayState,
  formatReplayState,
  formatPrompt,
  promptText,
  batchCounts,
  caseTitles,
} from './replay.js';

const REPLAY_URL = 'data/replay.json';
const WHICHES = ['base', 'edited'];
const STEP_TOKENS = TOKENS_PER_SECOND_1X; // arrow key = one second of 1x playback
const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** fetch + JSON.parse with byte progress (content-length may be absent when gzipped). */
async function fetchJsonWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) return res.json();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

/* ---------- pane text: diffed spans so streaming stays cheap ---------- */

function createPaneText(scroll) {
  const empty = scroll.querySelector('.rp-empty');
  const regions = { think: scroll.querySelector('.think'), answer: scroll.querySelector('.answer') };
  let rendered = []; // [{ region, cls, from, to, node }]

  const makeNode = (run, span) => {
    const text = textOf(run, span.from, span.to);
    if (!span.cls) return document.createTextNode(text);
    const el = document.createElement('span');
    el.className = span.cls;
    el.textContent = text;
    return el;
  };

  return {
    reset() {
      for (const r of rendered) r.node.remove();
      rendered = [];
    },
    render(run, cursor) {
      const spans = textSpans(run, cursor);
      let i = 0;
      for (; i < spans.length && i < rendered.length; i++) {
        const a = rendered[i];
        const b = spans[i];
        if (a.region !== b.region || a.cls !== b.cls || a.from !== b.from) break;
        if (a.to !== b.to) {
          if (b.to > a.to && a.node.nodeType === Node.TEXT_NODE) a.node.appendData(textOf(run, a.to, b.to));
          else a.node.textContent = textOf(run, b.from, b.to);
          a.to = b.to;
        }
      }
      for (const r of rendered.slice(i)) r.node.remove();
      rendered = rendered.slice(0, i);
      for (const span of spans.slice(i)) {
        const node = makeNode(run, span);
        regions[span.region].appendChild(node);
        rendered.push({ ...span, node });
      }
      const hasThink = rendered.some((r) => r.region === 'think');
      const hasAnswer = rendered.some((r) => r.region === 'answer');
      regions.think.hidden = !hasThink;
      regions.answer.hidden = !hasAnswer;
      empty.hidden = hasThink || hasAnswer;
    },
  };
}

/* ---------- confetti / sparkle burst ---------- */

const CONFETTI_COLORS = ['#ffd23f', '#2fc79a', '#8c6bff', '#4cb8ff', '#ff6f59'];

function burst(el) {
  if (reducedMotion()) return;
  const box = document.createElement('div');
  box.className = 'confetti';
  box.setAttribute('aria-hidden', 'true');
  for (let i = 0; i < 26; i++) {
    const p = document.createElement('i');
    const angle = (i / 26) * Math.PI * 2;
    const dist = 60 + (i % 5) * 18;
    p.style.setProperty('--dx', `${Math.cos(angle) * dist}px`);
    p.style.setProperty('--dy', `${Math.sin(angle) * dist - 30}px`);
    p.style.setProperty('--rot', `${(i * 47) % 360}deg`);
    p.style.background = CONFETTI_COLORS[i % CONFETTI_COLORS.length];
    box.appendChild(p);
  }
  el.appendChild(box);
  setTimeout(() => box.remove(), 1400);
}

/* ---------- the stage ---------- */

function createStage(data) {
  const cases = data.cases;
  const prepared = new Map();
  const getCase = (id) => {
    if (!prepared.has(id)) prepared.set(id, prepareCase(cases.find((c) => c.id === id)));
    return prepared.get(id);
  };

  const url = parseReplayState(location.search, cases);
  const state = {
    caseId: url.caseId,
    cursor: url.cursor ?? 0,
    speed: url.speed ?? DEFAULT_SPEED,
    playing: false,
    lastTs: 0,
    autoplayed: url.cursor != null, // a shared moment opens paused
    prevDone: { base: false, edited: false },
  };

  const panes = {};
  for (const which of WHICHES) {
    panes[which] = {
      scroll: $(`rp-scroll-${which}`),
      text: createPaneText($(`rp-scroll-${which}`)),
      finish: $(`rp-finish-${which}`),
      escape: $(`rp-escape-${which}`),
      meta: $(`rp-meta-${which}`),
      loop: $(`rp-loop-${which}`),
      loopLabel: $(`rp-looplabel-${which}`),
    };
  }

  // The neuron trace follows this cursor; scrubbing it seeks the replay.
  const neurons = createNeuronView({
    seek: (cursor) => {
      seek(cursor);
      writeUrl();
    },
    onLoad: () => render(),
  });

  const moodOf = (run) =>
    run.finish === 'stop' && run.correct ? 'win' : run.finish === 'stop' ? 'meh' : 'lose';

  /**
   * Lay the counter stickers out at their final size before playback and keep
   * them invisible (`.pre`). The strip then reserves exactly the space they will
   * fill, so revealing one never changes the height of the counters strip.
   */
  function primeStickers(c) {
    // Every label is measured across all cases and the element is pinned to the
    // tallest one: verdict strings differ in length, so a longer case would
    // otherwise wrap to a second line and grow the strip / the pill.
    const verdict = $('rp-verdict');
    verdict.classList.add('pre');
    verdict.hidden = false;
    let vh = 0;
    for (const other of cases) {
      const end = countersAt(other, replayLength(other)).verdict;
      verdict.textContent = end ? `Final: base  -  ${end.base}. Edited  -  ${end.edited}.` : '';
      vh = Math.max(vh, verdict.offsetHeight);
    }
    verdict.style.minHeight = `${vh}px`;

    for (const which of WHICHES) {
      const done = $(`ctr-done-${which}`);
      done.hidden = false; // the HTML ships it hidden; `.pre` reserves the slot
      let h = 0;
      for (const other of cases) {
        const run = other[which];
        done.textContent = runVerdict(run, capOf(other));
        done.className = `done-sticker ${moodOf(run)} pre`;
        h = Math.max(h, done.offsetHeight);
      }
      const run = c[which];
      done.textContent = runVerdict(run, capOf(c));
      done.className = `done-sticker ${moodOf(run)} pre`;
      done.style.minHeight = `${h}px`;
    }
    // Always reserve the loop row too, even for cases that never loop: the strip
    // must not change height when you pick a different case.
    const loop = $('ctr-loop');
    const repeats = Math.max(
      1,
      ...(c.base.loops ?? []).map((s) => countersAt(c, s.end).base.loop?.repeats ?? 0),
    );
    loop.textContent = `stuck in a loop ×${repeats}`;
    loop.className = 'loop-sticker pre';
    loop.hidden = false;
  }

  /**
   * Reserve the pane-head height using the longest finish label. The pill gains
   * text as the run progresses ("thinking…" → "⚠ hit token limit"), which on a
   * narrow pane wraps the head onto a second line and shoves the stream down.
   */
  function reserveHeadHeight() {
    for (const which of WHICHES) {
      const head = $(`rp-pane-${which}`).querySelector('.pane-head');
      const finish = $(`rp-finish-${which}`);
      head.style.minHeight = '';
      const text = finish.textContent;
      const cls = finish.className;
      let h = 0;
      for (const label of ['✓ stopped on its own', '⚠ hit token limit']) {
        finish.textContent = label;
        h = Math.max(h, head.offsetHeight);
      }
      finish.textContent = text;
      finish.className = cls;
      head.style.minHeight = `${h}px`;
    }
  }

  /* --- case picker + prompt --- */
  const caseButtons = new Map();
  function buildCases() {
    const root = $('rp-cases');
    root.innerHTML = '';
    const titles = caseTitles(cases);
    for (const c of cases) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = `case case-${c.kind}`;
      b.setAttribute('role', 'radio');
      const tag = document.createElement('span');
      tag.className = 'case-tag';
      tag.textContent = c.kind === 'control' ? '◎ control · easy' : '↻ loop case';
      const title = document.createElement('span');
      title.className = 'case-title';
      title.textContent = titles.get(c.id);
      const snip = document.createElement('span');
      snip.className = 'case-snip';
      const text = promptText(c.prompt);
      snip.textContent = text.length > 84 ? text.slice(0, 83).trimEnd() + '…' : text;
      const foot = document.createElement('span');
      foot.className = 'case-foot';
      const out = (r) => (r.finish === 'length' ? '⚠ cap' : `${r.correct ? '✓' : '✗'} ${fmt(r.nTokens)}`);
      const fb = document.createElement('b');
      fb.className = 'c-base';
      fb.textContent = `base ${out(c.base)}`;
      const fe = document.createElement('b');
      fe.className = 'c-edited';
      fe.textContent = `edited ${out(c.edited)}`;
      foot.append(fb, fe);
      b.append(tag, title, snip, foot);
      b.onclick = () => selectCase(c.id, { play: true });
      root.appendChild(b);
      caseButtons.set(c.id, b);
    }
  }

  function paintPrompt(c, p, metaEl) {
    p.innerHTML = '';
    const { segments, hasInstruction } = formatPrompt(c.prompt);
    for (const s of segments) {
      if (s.math) {
        const m = document.createElement('span');
        m.className = 'math';
        m.textContent = s.text;
        p.appendChild(m);
      } else p.appendChild(document.createTextNode(s.text));
    }
    const sm = c.sampling ?? {};
    const parts = [
      `answer key: ${c.reference}`,
      hasInstruction ? 'plus “reason step by step, put the answer in \\boxed{}”' : null,
      `temp ${sm.temperature} · top-p ${sm.top_p} · top-k ${sm.top_k} · presence penalty ${sm.presence_penalty} · cap ${fmt(sm.max_tokens)} · seed ${c.seed}`,
    ].filter(Boolean);
    let meta = parts.join(' · ');
    if (c.setting !== data.source.setting) {
      meta += `  -  recorded in the “${c.setting}” batch (presence penalty ${sm.presence_penalty}), because at “${data.source.setting}” the base model never stops on its own.`;
    }
    metaEl.textContent = meta;
  }

  function renderPrompt(c) {
    paintPrompt(c, $('rp-prompt'), $('rp-prompt-meta'));
  }

  /**
   * Reserve the tallest prompt and meta line across every case. Prompts differ
   * in length, so without this the player card resizes when you pick another
   * case and everything below it jumps. Re-run on resize: wrapping is
   * width-dependent.
   */
  function reservePromptHeight() {
    const p = $('rp-prompt');
    const meta = $('rp-prompt-meta');
    p.style.minHeight = '';
    meta.style.minHeight = '';
    let hP = 0;
    let hM = 0;
    for (const c of cases) {
      paintPrompt(c, p, meta);
      hP = Math.max(hP, p.offsetHeight);
      hM = Math.max(hM, meta.offsetHeight);
    }
    p.style.minHeight = `${hP}px`;
    meta.style.minHeight = `${hM}px`;
    renderPrompt(getCase(state.caseId));
  }

  /* --- transport --- */
  function buildSpeeds() {
    const root = $('rp-speeds');
    root.innerHTML = '';
    for (const s of SPEEDS) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'speed-btn';
      b.dataset.speed = String(s);
      b.textContent = `${s}×`;
      b.title = `${s * TOKENS_PER_SECOND_1X} tokens per second`;
      b.onclick = () => {
        state.speed = s;
        writeUrl();
        render();
      };
      root.appendChild(b);
    }
  }

  function selectCase(id, { play = false } = {}) {
    state.caseId = id;
    state.cursor = 0;
    state.prevDone = { base: false, edited: false };
    for (const which of WHICHES) {
      panes[which].text.reset();
      panes[which].scroll.scrollTop = 0;
    }
    renderPrompt(getCase(id));
    primeStickers(getCase(id));
    writeUrl();
    if (play) setPlaying(true);
    render({ follow: true });
  }

  function setPlaying(on) {
    const c = getCase(state.caseId);
    if (on && state.cursor >= replayLength(c)) {
      state.cursor = 0;
      state.prevDone = { base: false, edited: false };
    }
    state.playing = on;
    state.autoplayed = true;
    state.lastTs = 0;
    if (on) requestAnimationFrame(tick);
    else writeUrl();
    render();
  }

  function seek(cursor, { pause = true } = {}) {
    const c = getCase(state.caseId);
    if (pause && state.playing) {
      state.playing = false;
      state.autoplayed = true;
    }
    state.cursor = clampCursor(cursor, replayLength(c));
    render({ follow: true });
  }

  function tick(ts) {
    if (!state.playing) return;
    const c = getCase(state.caseId);
    const end = replayLength(c);
    const dt = state.lastTs ? ts - state.lastTs : 0;
    state.lastTs = ts;
    state.cursor = advanceCursor(state.cursor, dt, state.speed, end);
    if (state.cursor >= end) {
      state.playing = false;
      writeUrl();
    }
    render();
    if (state.playing) requestAnimationFrame(tick);
  }

  let urlTimer = 0;
  function writeUrl() {
    clearTimeout(urlTimer);
    urlTimer = setTimeout(() => {
      const params = new URLSearchParams(location.search);
      for (const k of ['case', 't', 'speed']) params.delete(k);
      const own = new URLSearchParams(
        formatReplayState({ caseId: state.caseId, cursor: state.playing ? null : state.cursor, speed: state.speed }),
      );
      for (const [k, v] of own) params.set(k, v);
      const qs = params.toString();
      history.replaceState(null, '', `${location.pathname}${qs ? '?' + qs : ''}${location.hash}`);
    }, 120);
  }

  /* --- render --- */
  function render({ follow = false } = {}) {
    const c = getCase(state.caseId);
    const end = replayLength(c);
    const cap = capOf(c);
    const k = countersAt(c, state.cursor);

    for (const [id, b] of caseButtons) {
      const on = id === state.caseId;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    }
    for (const b of $('rp-speeds').children) {
      b.setAttribute('aria-pressed', String(Number(b.dataset.speed) === state.speed));
    }
    const playBtn = $('rp-play');
    playBtn.textContent = state.playing ? '❚❚ Pause' : state.cursor >= end ? '↺ Replay' : '▶ Play';
    playBtn.setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
    $('rp-clock').textContent = `token ${fmt(k.cursor)} / ${fmt(end)} · ${state.speed}× = ${fmt(state.speed * TOKENS_PER_SECOND_1X)} tok/s`;

    // panes
    for (const which of WHICHES) {
      const run = c[which];
      const st = k[which];
      const pane = panes[which];
      const stick = follow || isNearBottom(pane.scroll);
      pane.text.render(run, state.cursor);

      pane.finish.className = 'finish' + (st.done ? (st.finish === 'stop' ? ' good' : ' bad') : '');
      pane.finish.textContent = st.done
        ? st.finish === 'stop'
          ? '✓ stopped on its own'
          : '⚠ hit token limit'
        : st.spent === 0
          ? 'waiting'
          : st.loop
            ? '↻ looping…'
            : st.phase === 'answer'
              ? 'answering…'
              : 'thinking…';

      const rep = repetitionAt(run, state.cursor);
      drawLoopMeter(pane.loop, rep, which);
      pane.loopLabel.textContent = `${Math.round(rep * 100)}%`;
      pane.meta.textContent =
        `${fmt(st.spent)} / ${fmt(run.nTokens)} tokens` +
        (run.thinkEnd != null ? ` · </think> at ${fmt(run.thinkEnd)}` : ' · never left thinking') +
        (st.loopTokens ? ` · ${fmt(st.loopTokens)} in loops` : '');

      // finish: a bar under the pane text + a sticker in the counters strip
      // (with a burst when the run reaches its end during playback)
      const mood = moodOf(run);
      const sticker = $(`ctr-done-${which}`);
      pane.escape.hidden = !st.done;
      if (st.done) {
        pane.escape.textContent = runVerdict(run, cap);
        pane.escape.className = `escape ${mood}`;
        if (!state.prevDone[which] && state.playing && mood === 'win') burst(sticker);
      }
      // The sticker's text and size were laid out up front (primeStickers); only
      // its visibility changes here, so the counters strip keeps its height.
      sticker.classList.toggle('pre', !st.done);
      state.prevDone[which] = st.done;

      // Pin to the bottom *after* the banner toggles: the banner takes its
      // height out of the stream, so a scrollTop set before it appeared would
      // leave the last line clipped behind it.
      if (stick) pane.scroll.scrollTop = pane.scroll.scrollHeight;
    }
    const doneSub = (run) => (run.finish === 'stop' ? 'stopped on its own' : 'out of budget');

    // counters + mascots
    const b = k.base;
    $('ctr-base-num').textContent = fmt(b.spent);
    $('ctr-base-sub').textContent = `of ${fmt(c.base.nTokens)} · ${
      b.done ? doneSub(c.base) : b.loop ? 'looping…' : b.phase === 'answer' ? 'answering' : 'thinking'
    }`;
    const loopEl = $('ctr-loop');
    const bubble = $('ctr-bubble');
    const ctrBase = $('ctr-base');
    const activeLoop = b.loop && b.loop.repeats >= 1 ? b.loop : null;
    ctrBase.classList.toggle('dizzy', Boolean(b.loop));
    // Reserved slot (see primeStickers): reveal, never add or remove layout.
    loopEl.classList.toggle('pre', !activeLoop);
    bubble.hidden = !activeLoop;
    if (activeLoop) {
      loopEl.textContent = `stuck in a loop ×${activeLoop.repeats}`;
      loopEl.title = `the same ${activeLoop.segment.period}-token passage, repeated verbatim`;
      const phrase = `“${loopPhrase(c.base, activeLoop.segment, 64)}”`;
      const bubbleText = $('ctr-bubble-text');
      if (bubbleText.textContent !== phrase) bubbleText.textContent = phrase;
      bubble.title = phrase;
    }
    ctrBase.classList.toggle('done', b.done);
    neurons.update(c, state.cursor);
    ctrBase.classList.toggle('zap', neurons.spikeAt(state.cursor));

    const e = k.edited;
    $('ctr-edited-num').textContent = fmt(e.spent);
    $('ctr-edited-sub').textContent = `of ${fmt(c.edited.nTokens)} · ${
      e.done ? doneSub(c.edited) : e.loop ? 'looping…' : e.phase === 'answer' ? 'answering' : 'thinking'
    }`;
    const progress = Math.min(1, e.spent / Math.max(1, c.edited.nTokens));
    $('ctr-snake').setAttribute('transform', `translate(${46 + progress * 166} 0)`);
    $('ctr-edited').classList.toggle('home', e.done && c.edited.finish === 'stop');

    $('ctr-waste-num').textContent = fmt(k.wastedLooping);
    $('ctr-waste-sub').textContent =
      `base ${fmt(b.loopTokens)} · edited ${fmt(e.loopTokens)} tokens inside verbatim repeats`;

    // Pre-filled and pre-sized by primeStickers: reveal only, never reflow.
    $('rp-verdict').classList.toggle('pre', !k.verdict);

    // timeline / scrubber
    const lane = (run) => ({ n: run.nTokens, thinkEnd: run.thinkEnd, finish: run.finish, loops: run.loops });
    drawTimeline($('rp-timeline'), lane(c.base), lane(c.edited), { cursor: state.cursor, maxN: Math.max(end, cap ?? 0), cap });
    const scrub = $('rp-scrub');
    scrub.setAttribute('aria-valuemax', String(end));
    scrub.setAttribute('aria-valuenow', String(k.cursor));
    scrub.setAttribute('aria-valuetext', `token ${fmt(k.cursor)} of ${fmt(end)}`);
  }

  /* --- scrubber input --- */
  function cursorFromPointer(ev) {
    const canvas = $('rp-timeline');
    const rect = canvas.getBoundingClientRect();
    const { padL, plotW } = timelineGeometry(rect.width);
    const c = getCase(state.caseId);
    const maxN = Math.max(replayLength(c), capOf(c) ?? 0);
    return ((ev.clientX - rect.left - padL) / plotW) * maxN;
  }

  function wireScrubber() {
    const scrub = $('rp-scrub');
    let dragging = false;
    scrub.addEventListener('pointerdown', (ev) => {
      dragging = true;
      scrub.setPointerCapture(ev.pointerId);
      seek(cursorFromPointer(ev));
    });
    scrub.addEventListener('pointermove', (ev) => {
      if (dragging) seek(cursorFromPointer(ev));
    });
    const stop = () => {
      if (!dragging) return;
      dragging = false;
      writeUrl();
    };
    scrub.addEventListener('pointerup', stop);
    scrub.addEventListener('pointercancel', stop);
    scrub.addEventListener('keydown', (ev) => {
      const c = getCase(state.caseId);
      const step = ev.shiftKey ? STEP_TOKENS * 10 : STEP_TOKENS;
      const moves = {
        ArrowRight: state.cursor + step,
        ArrowUp: state.cursor + step,
        ArrowLeft: state.cursor - step,
        ArrowDown: state.cursor - step,
        PageUp: state.cursor + STEP_TOKENS * 20,
        PageDown: state.cursor - STEP_TOKENS * 20,
        Home: 0,
        End: replayLength(c),
      };
      if (ev.key in moves) {
        ev.preventDefault();
        seek(Math.round(moves[ev.key]));
        writeUrl();
      } else if (ev.key === ' ' || ev.key === 'Enter') {
        ev.preventDefault();
        setPlaying(!state.playing);
      }
    });
  }

  function wireTransport() {
    $('rp-play').onclick = () => setPlaying(!state.playing);
    $('rp-skip').onclick = () => {
      seek(replayLength(getCase(state.caseId)));
      writeUrl();
    };
    $('rp-restart').onclick = () => {
      seek(0, { pause: false });
      if (!state.playing) setPlaying(true);
    };
    $('rp-share').onclick = async () => {
      const btn = $('rp-share');
      const params = new URLSearchParams(location.search);
      for (const [k, v] of new URLSearchParams(
        formatReplayState({ caseId: state.caseId, cursor: state.cursor, speed: state.speed }),
      )) params.set(k, v);
      const link = `${location.origin}${location.pathname}?${params}`;
      try {
        await navigator.clipboard.writeText(link);
        btn.textContent = '✓ link copied';
      } catch {
        btn.textContent = '✕ copy failed';
      }
      setTimeout(() => (btn.textContent = 'copy link'), 1600);
    };
  }

  /**
   * Autoplay once the player or the counters strip below the panes is mostly on
   * screen, so the loop and escape moments play where the visitor is looking
   * (not while they read the hero); a shared ?t= moment opens paused.
   */
  function wireAutoplay() {
    if (!('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting) && !state.autoplayed) {
          io.disconnect();
          setPlaying(true);
        }
      },
      { threshold: 0.6 },
    );
    io.observe($('rp-player'));
    io.observe($('rp-counters'));
  }

  buildCases();
  buildSpeeds();
  wireTransport();
  wireScrubber();
  reservePromptHeight(); // also paints the current prompt
  reserveHeadHeight();
  primeStickers(getCase(state.caseId));
  // Web fonts change line wrapping: re-measure once they are in.
  document.fonts?.ready.then(() => {
    reservePromptHeight();
    reserveHeadHeight();
    neurons.resize();
    render();
  });

  return {
    start() {
      render({ follow: true });
      wireAutoplay();
      // A shared link jumps straight to the player (prompt, scrubber, counters).
      if (url.fromUrl) $('rp-player').scrollIntoView({ block: 'start', behavior: 'instant' });
    },
    redraw: () => {
      reservePromptHeight();
      reserveHeadHeight();
      neurons.resize();
      render();
    },
  };
}

/* ---------- honest data card ---------- */

function dotRow(label, cls, groups) {
  const row = document.createElement('div');
  row.className = 'dot-row';
  const name = document.createElement('span');
  name.className = `dot-name ${cls}`;
  name.textContent = label;
  const dots = document.createElement('div');
  dots.className = 'dots';
  for (const g of groups) {
    for (let i = 0; i < g.count; i++) {
      const d = document.createElement('i');
      d.className = `pdot ${g.cls}`;
      d.title = g.title;
      dots.appendChild(d);
    }
  }
  row.append(name, dots);
  return row;
}

function bigStat(num, label, cls) {
  const el = document.createElement('div');
  el.className = 'hstat';
  const n = document.createElement('span');
  n.className = `hnum ${cls}`;
  n.textContent = num;
  const l = document.createElement('span');
  l.className = 'hlabel';
  l.textContent = label;
  el.append(n, l);
  return el;
}

const pct = (x) => `${(x * 100).toFixed(1).replace(/\.0$/, '')}%`;

function renderHonestCard(data) {
  const badge = $('honest-badge');
  badge.textContent = 'recorded';
  badge.className = 'badge pre';
  const s = data.batchStats;
  const k = batchCounts(s);
  const sm = data.source.sampling;
  $('honest-lede').textContent =
    `${k.n} MATH-500 problems, both models, cap ${fmt(sm.max_tokens)} tokens, temp ${sm.temperature}, ` +
    `presence penalty ${sm.presence_penalty}, seed ${data.source.seed}. Each dot is one problem.`;

  const body = $('honest-body');
  body.innerHTML = '';

  const grid = document.createElement('div');
  grid.className = 'honest-grid';

  const left = document.createElement('div');
  left.className = 'honest-dots';
  left.append(
    dotRow('base', 'c-base', [
      { count: k.base.correct, cls: 'win', title: 'stopped, correct' },
      { count: k.base.stoppedWrong, cls: 'meh', title: 'stopped, wrong answer' },
      { count: k.base.cap, cls: 'cap', title: 'hit the token cap' },
    ]),
    dotRow('edited', 'c-edited', [
      { count: k.edited.correct, cls: 'win', title: 'stopped, correct' },
      { count: k.edited.stoppedWrong, cls: 'meh', title: 'stopped, wrong answer' },
      { count: k.edited.cap, cls: 'cap', title: 'hit the token cap' },
    ]),
  );
  const legend = document.createElement('div');
  legend.className = 'dot-legend';
  legend.innerHTML =
    '<span><i class="pdot win"></i>stopped &amp; correct</span><span><i class="pdot meh"></i>stopped, wrong</span><span><i class="pdot cap"></i>hit the cap</span>';
  left.appendChild(legend);

  const right = document.createElement('div');
  right.className = 'honest-stats';
  right.append(
    bigStat(`${k.base.cap}/${k.n}`, `base hit the cap · ${pct(s.base_acc)} correct`, 'c-base'),
    bigStat(`${k.edited.cap}/${k.n}`, `edited hit the cap · ${pct(s.edited_acc)} correct`, 'c-edited'),
  );
  grid.append(left, right);
  body.appendChild(grid);

  const notes = document.createElement('div');
  notes.className = 'honest-notes';
  const note = (title, text, cls = '') => {
    const n = document.createElement('div');
    n.className = `hnote ${cls}`;
    const h = document.createElement('h4');
    h.textContent = title;
    const p = document.createElement('p');
    p.textContent = text;
    n.append(h, p);
    notes.appendChild(n);
    return n;
  };

  const loopCases = data.cases.filter((c) => c.kind === 'loop').length;
  note(
    'How the replay cases were picked',
    `Loop cases are problems where base hits the cap and edited stops with the right answer  -  that happened ` +
      `${k.edited.correct} times out of ${k.n}; the replay shows ${loopCases} of them. The dots above are the whole batch.`,
  );

  const control = data.cases.find((c) => c.kind === 'control');
  if (control) {
    note(
      'Checking for regressions',
      `On “${promptText(control.prompt)}” both models stop with ${control.reference}: base in ` +
        `${fmt(control.base.nTokens)} tokens, edited in ${fmt(control.edited.nTokens)}. ` +
        'This provides a small sanity check; broader evaluation would be needed to assess effects on other capabilities. ' +
        'Pick “◎ control” above to replay it.',
      'ok',
    );
  }
  body.appendChild(notes);
}

function renderLoadError(err) {
  $('rp-loading-title').textContent = 'The recording didn’t load';
  $('rp-loading-text').textContent =
    `Couldn’t fetch ${REPLAY_URL} (${err instanceof Error ? err.message : String(err)}). ` +
    'You can still run both models live further down.';
  $('rp-loading').classList.add('failed');
  const badge = $('honest-badge');
  badge.textContent = 'not loaded';
  renderNeuronLoadError();
}

/**
 * Mount the replay stage: fetch replay.json after first paint, then build the
 * case picker, player, counters and the honest data card. Returns a redraw
 * hook for window resizes.
 */
export function mountReplay() {
  let stage = null;
  const load = async () => {
    try {
      const data = await fetchJsonWithProgress(REPLAY_URL, (loaded, total) => {
        const fill = $('rp-loading-fill');
        const known = total > 0;
        fill.style.width = known ? `${Math.min(100, (loaded / total) * 100)}%` : '100%';
        fill.classList.toggle('indeterminate', !known);
        $('rp-loading-text').textContent = `Fetching the recorded token streams… ${(loaded / 1024 / 1024).toFixed(2)} MB` +
          (known ? ` of ${(total / 1024 / 1024).toFixed(2)} MB` : '');
      });
      if (!data?.cases?.length || data.placeholder) throw new Error('no recorded cases');
      $('rp-loading').hidden = true;
      $('rp-stage').hidden = false;
      stage = createStage(data);
      stage.start();
      renderHonestCard(data);
    } catch (err) {
      console.error(err);
      renderLoadError(err);
    }
  };
  // Let the page paint first; the recording is not needed for first paint.
  requestAnimationFrame(() => setTimeout(load, 0));
  return () => stage?.redraw();
}
