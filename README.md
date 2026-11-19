# Weight-Space Playground

An in-browser showcase of **weight-space programming**: an LLM edited 0.016% of
[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)'s weights — no training — and the
model stopped getting stuck in reasoning loops. The page replays a real recording of the base and
edited models side by side, then lets you run both locally in your browser with WebGPU.

**Live:** <https://tabbyml.github.io/weight-space-playground/>

Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Exp 8, MIT CSAIL). Built by the [TabbyML team](https://tabbyml.com).

## What it shows

- **Recorded replay** — same prompt, same seed, base vs edited, with a token timeline, loop meter,
  and the exact weight-space diff.
- **Key neurons** — the recorded activation projection onto the subtracted direction, layer by layer.
- **Live run** — both 4-bit GGUFs run locally via [wllama](https://github.com/ngxson/wllama)
  (llama.cpp → WebAssembly + WebGPU). Nothing is sent to a server; the first run downloads ~1.05 GB
  once and caches it in the browser.
