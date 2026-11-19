# Offline precompute pipeline (M1 "story without download")

Produces the real data behind the showcase's recorded replay and weight-diff panel.
Large inputs, builds and raw recordings live under `~/.cache/weight-editing/`
(override with `WE_CACHE`); only small outputs land in the repo:

| Output | Size | What |
|---|---|---|
| `playground/public/data/weight-diff.json` | ~30 KB | exact bf16 edit: columns, per-column delta, rank structure, GGUF survival |
| `playground/public/data/replay.json` | see `validate` | chosen demo cases with full token streams + loop segments |
| `tools/precompute/out/batch-summary.json` | ~50 KB | every candidate × setting × model: tokens, finish, correct |
| `playground/public/data/neuron-trace/` | ~1.4 MB | `index.json` + one `.bin` per replay case: per-token activity of the 64 edited neurons (bf16 trace, int8-quantised) |
| `tools/precompute/out/neuron-findings.md` | ~19 KB | analysis of the traces (loop vs non-loop, onset, counterfactual, stop, channels) |

Schemas: `playground/public/data/README.md`. All remote inputs are pinned in `config.py`.

## Setup

```bash
python3 -m venv ~/.cache/weight-editing/venv
~/.cache/weight-editing/venv/bin/pip install -r tools/precompute/requirements.txt
# llama.cpp at the pinned commit (b12211 = 7d7f02875)
git clone https://github.com/ggml-org/llama.cpp ~/.cache/weight-editing/llama.cpp
git -C ~/.cache/weight-editing/llama.cpp checkout 7d7f0287530bc8981690f361d8c1627dbae1c99a
# CPU build (used for the page4096 batch)
cmake -S ~/.cache/weight-editing/llama.cpp -B ~/.cache/weight-editing/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.cache/weight-editing/llama.cpp/build -j 48 --target llama-server
# GPU neuron trace (step 6): torch cu128 wheels + transformers (no nvcc needed)
~/.cache/weight-editing/venv/bin/pip install -r tools/precompute/requirements-trace.txt
# GPU: official Vulkan release of the same build (no CUDA toolkit needed; used for the 16k batches)
mkdir -p ~/.cache/weight-editing/llama-vulkan && cd ~/.cache/weight-editing/llama-vulkan \
  && curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b12211/llama-b12211-bin-ubuntu-vulkan-x64.tar.gz \
  && tar xzf llama-b12211-bin-ubuntu-vulkan-x64.tar.gz
```

## Steps (run from the repo root with the venv's python; each is idempotent/cached)

```bash
PY=~/.cache/weight-editing/venv/bin/python
$PY -m tools.precompute.fetch                 # 1. MATH-500, both GGUFs, both bf16 safetensors
$PY -m tools.precompute.weight_diff           # 2. -> data/weight-diff.json (~2 min CPU)
# 3. record (skips runs already on disk)
$PY -m tools.precompute.record --settings page4096 --servers-base 8 --servers-edited 8 --threads 4   # CPU
$PY -m tools.precompute.record --backend vulkan --settings up16k_pp cap16k \
    --gpus 0 1 2 3 --servers-base 8 --servers-edited 8 --slots 4                                    # GPU
$PY -m tools.precompute.build_replay          # 4+5. -> out/batch-summary.json, data/replay.json
$PY -m tools.precompute.neuron_trace --gpu 0  # 6. bf16 traces (~25 s on one RTX 4090) -> data/neuron-trace/
$PY -m tools.precompute.neuron_findings       # 7. -> out/neuron-findings.md (keeps the hand-written head)
$PY -m tools.precompute.validate              # schema/consistency check of all outputs
$PY -m pytest -q tools/precompute/tests       # unit tests of the pure helpers
```

| Module | Role |
|---|---|
| `config.py` | pins (HF revisions, GGUF URLs = page `MODELS`, llama.cpp build), paths, recording settings |
| `fetch.py` | downloads into the cache |
| `weight_diff.py` | byte-compares all 488 bf16 tensors; summarises the differing ones (SVD, edit form, GGUF survival) |
| `candidates.py` | default prompt + deterministic difficulty-stratified MATH-500 sample (integer answers) |
| `record.py` | runs llama-server (CPU or Vulkan), one `/completion` per (setting, model, case, seed); raw JSON per run |
| `tokens.py` | response → exact token pieces, `</think>` index, finish reason, graded answer |
| `answers.py` | `\boxed{}` extraction + normalisation + grading |
| `loops.py` | verbatim-loop detection (periodic spans) |
| `build_replay.py` | batch summary for every setting, case selection, `replay.json` |
| `neuron_trace.py` | bf16 teacher-forced forward (transformers, GPU) per replay case × {base, edited, counterfactual}; hooks `mlp.down_proj` at layers 15/19; cache + quantised served files |
| `neuron_findings.py` | analysis of the full-precision trace caches -> `out/neuron-findings.md` |
| `validate.py` | dependency-free schema checks (incl. the neuron-trace binaries: layout, bins, int8 round-trip, cf/base consistency) |

## Recording settings (`config.SETTINGS`)

| Setting | temp / top_p / top_k | presence_penalty | max_tokens | n_ctx | why |
|---|---|---|---|---|---|
| `page4096` | 0.6 / 0.95 / 20 | 0 | 4096 | 8192 | the page's current live defaults |
| `up16k_pp` | 0.6 / 0.95 / 20 | 1.5 | 16384 | 20480 | upstream exp8 eval used presence_penalty 1.5 (and 32k cap) |
| `cap16k` | 0.6 / 0.95 / 20 | 0 | 16384 | 20480 | separates the cap effect from the penalty effect |

Other samplers are llama-server defaults (min_p 0.05, repeat_penalty 1.0, no DRY/XTC);
wllama's chat completion runs the same server code, so the page gets the same defaults.
Prompts are rendered by each GGUF's own chat template with `enable_thinking: true`.
Seed 0 everywhere.

**Reproducibility.** CPU runs are one sequence per server with a fixed thread count.
GPU runs use 4 parallel slots per server (continuous batching), so a rerun may diverge
token-for-token; the raw responses are kept in the cache and each records its backend.
None of these runs reproduce a browser WebGPU run token-for-token.

## Case selection (`build_replay.py`)

`loop` = base `finish == "length"` and edited `"stop"` and correct. Order: the default
prompt first if it qualifies, then cases whose base run has a verbatim loop segment, then
by edited length ascending; max 4. `control` = both stop and are correct, fewest total
tokens.

**Replay setting = `cap16k`** (`config.REPLAY_SETTING`, operator decision 2026-11-19):

| Setting | base cap | edited cap | base acc | edited acc | loop cases | default prompt |
|---|---|---|---|---|---|---|
| `page4096` (CPU) | 100% | 100% | 0% | 0% | 0 | both cap |
| `up16k_pp` (Vulkan) | 90.3% | 80.6% | 9.7% | 12.9% | 2 | **reversed**: base solves, edited caps |
| `cap16k` (Vulkan) | 100% | 54.8% | 0% | 32.3% | 10 | both cap |

`cap16k` is the page's samplers with a larger cap and gives the strongest honest
contrast; the upstream-faithful `up16k_pp` (presence_penalty 1.5) was weak and reversed
the default prompt. Base never stops in `cap16k`, so the `control` case is taken from
`up16k_pp` (`CONTROL_FALLBACK`) and labelled with that setting's sampling. All three
batches' stats are in `replay.json` (`batchStats` + `otherSettings` + `settingNote`) and
in `out/batch-summary.json`. Override with `--replay-setting`.

**Token streams** are rebuilt from the response's token ids with the GGUF vocabulary
(`vocab-<model>.json` cached), not from `completion_probabilities`: llama-server merges
tokens that only together form a UTF-8 character into one probability entry, so that list
can be shorter. Counts are checked against `tokens_predicted` and the joined text against
`content`; logprobs are aligned by id (null for merged tokens).

## Neuron trace (`neuron_trace.py`, `neuron_findings.py`)

- **Backend:** transformers 5.23.0 + torch 2.11.0+cu128 on one RTX 4090, bf16, one
  teacher-forced forward per run (`use_cache=False`, whole sequence at once, ≤ 16.5k tokens,
  0.2–1.8 s each). Qwen3.5's Gated DeltaNet layers run on transformers' reference PyTorch
  kernels (flash-linear-attention / causal_conv1d not installed); the forward is validated by
  the bf16 top-1 agreement with the recorded tokens (within 1.5 pp of the Q4_K_M recording's own
  rate, see `out/neuron-findings.md`). No llama.cpp fallback was needed.
- **Weights:** the base checkpoint is loaded once. Edited runs copy the two edited bf16
  `down_proj` tensors in place; `weight-diff.json` asserts these are the only differing tensors.
- **Tokens:** prompt ids from `llama-tokenize` (the Vulkan release, same tokenizer as
  llama-server) are asserted equal to the HF tokenizer's ids and to `tokens_evaluated`. Generated
  ids are the recorded ids, asserted to match the GGUF vocab entry by entry and to decode to the
  recorded text exactly. Re-encoding the text would *not* always reproduce them (2 of 10 streams
  were sampled with a non-canonical split), which is why the recorded ids are fed directly.
- **Runs:** `base` (base weights, base tokens), `edited`, `counterfactual` (edited weights over
  the base tokens; really run). Cache: `~/.cache/weight-editing/neuron-trace/<setting>/<case>__<run>.{npz,json}`
  holds the full-precision `a[64]`, `proj`, and the bf16 log-prob of every recorded token, plus metadata.
  Pass `--force` to recompute and `--no-compute` to rebuild only the served files.
