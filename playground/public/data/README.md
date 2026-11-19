# Precomputed showcase data

The showcase's panels read small JSON files from this directory. They are produced
offline by `tools/precompute/` (see its README; every step is reproducible and pinned).
When a file is missing the page falls back to a clearly-labelled **placeholder**.

| File | Produced by | Source of truth |
|---|---|---|
| `weight-diff.json` | `python -m tools.precompute.weight_diff` | bf16 safetensors (base vs edited) |
| `replay.json` | `python -m tools.precompute.build_replay` | native llama.cpp recordings on the pinned GGUFs |
| `../../../tools/precompute/out/batch-summary.json` (not served) | same | every recorded prompt × setting |
| `neuron-trace/index.json` + `neuron-trace/<case-id>.bin` | `python -m tools.precompute.neuron_trace` | bf16 teacher-forced forward (transformers) over the `replay.json` token streams; **quantised** (int8/uint8) |

Run `python -m tools.precompute.validate` after regenerating; it checks the schemas below.

## Conventions (read this first)

- Shapes use the **HF/PyTorch `[out, in]`** convention. `mlp.down_proj.weight` is
  `[1024, 3584]` = `[hidden, intermediate]` and computes `y = W @ x`, where `x` is the
  3584-d intermediate (post-SwiGLU) activation.
- The edit changes **64 columns** of each of the two tensors. A column `j` is an
  **input** column of `down_proj`, i.e. one **intermediate (MLP) neuron**: the 1024-d
  vector that neuron writes into the residual stream. Rows are *not* selectively
  changed: within those 64 columns all 1024 rows change in layer 15 and 1023 in layer 19.
- In GGUF the same tensor is `blk.{15,19}.ffn_down.weight` with ggml
  `ne = [3584, 1024]` (`ne[0]` is the contiguous input dimension). Same matrix, reversed
  dimension order.
- The weight diff comes from **bf16**, not from the GGUFs: the two pinned GGUFs are not
  a matched quant pair (base = unsloth imatrix Q4_K_M, edited = plain Q4_K_M), so a
  GGUF-vs-GGUF diff shows nearly every tensor changed.

## `weight-diff.json`

```jsonc
{
  "placeholder": false,
  "source": "bf16 safetensors",
  "quant": "bf16",                       // kept for the existing renderer's note line
  "base":   { "repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc0636…" },
  "edited": { "repo": "RikiyaT/Qwen3.5-0.8B-weight-space-programming", "revision": "d1671a1…" },
  "ggufPair": { "base": "<url>", "edited": "<url>" },   // the page's MODELS URLs
  "convention": "<one-line restatement of the conventions above>",
  "tensorsCompared": 488,               // every tensor in the checkpoint, byte-compared
  "tensors": [                          // only tensors that differ (layers 15 and 19)
    {
      "name": "model.language_model.layers.15.mlp.down_proj.weight",
      "ggufName": "blk.15.ffn_down.weight",
      "layer": 15,
      "shape": [1024, 3584],            // HF [out, in]
      "changedAxis": "columns",
      "changedRows": 1024,              // rows with any non-zero delta (not selective)
      "changedColumns": [0, 2, 11, …],  // 64 sorted ints: intermediate neuron ids
      "deltaNorm":      [ … ],          // 64: L2 of delta[:, j] per changed column
      "baseColumnNorm": [ … ],          // 64: L2 of base[:, j] (for relative scale)
      "entriesChanged": 65260,          // non-zero bf16 deltas (64*1024 nominal; a few round to 0)
      "totalEntries": 3670016,
      "pctChanged": 1.7782,             // percent of this tensor
      "maxAbsDelta": 0.146362,
      "frobeniusDelta": 1.74889,
      "frobeniusBase": 16.44282,
      "svd": {                          // SVD of the [1024, 64] delta sub-matrix
        "singularValues": [1.7489, 0.00099, …],  // top 8, descending
        "rank1EnergyFraction": 0.999995,          // s1^2 / sum s^2
        "rank1RelResidual": 0.0023,               // ||delta - u coeff^T|| / ||delta||
        "u": [ … ],                     // 1024: unit direction in the residual stream
        "coeff": [ … ]                  // 64: delta[:, changedColumns[k]] ≈ u * coeff[k]
      },
      "editForm": "w_j' = w_j - alpha * (u . w_j) * u",
      "alphaMean": 6.0006, "alphaStd": 0.0018,    // alpha_k = -coeff[k] / projBefore[k]
      "projBefore": [ … ],              // 64: u · base[:, j]
      "projAfter":  [ … ],              // 64: u · edited[:, j]  (≈ (1 - alpha) * projBefore)
      "gguf": {                         // optional: does the edit survive Q4_K_M?
        "ne": [3584, 1024],             // ggml order
        "types": { "base": "Q4_K", "edited": "Q4_K" },
        "cosine": 0.99267,              // cos(bf16 delta, GGUF delta) on the 64 columns
        "normRatio": 1.00342,           // ||GGUF delta|| / ||bf16 delta|| on the 64 columns
        "projOnURatio": 0.99606,        // GGUF delta's u-projection vs bf16's
        "unchangedColumnsRelDelta": 0.0755  // GGUF noise on the OTHER 3520 columns (quant mismatch)
      }
    }
    // layer 19 …
  ]
}
```

The sign of `u` is arbitrary (SVD); it is fixed so that at least half of the `coeff` are
negative (34/64 in layer 15, 40/64 in layer 19), a weak tie-break. The edit itself does
not depend on the sign. Floats in vectors are rounded through float16.

## `replay.json`

Recorded base vs edited runs for a few demo prompts, chosen from an offline batch
(`tools/precompute/out/batch-summary.json` has the whole batch for every setting).

```jsonc
{
  "placeholder": false,
  "recorded": true,
  "source": {
    "engine": "llama.cpp llama-server (native)",
    "llamaCpp": { "commit": "7d7f028…", "version": "version: 0.9.0-dev (build 12211, commit 7d7f02875)" },
    "backend": "Vulkan, RTX 4090 (…)",          // or "CPU (…)"
    "host": "<cpu model>; …",
    "gguf": { "base": "<url>", "edited": "<url>" },
    "setting": "cap16k",                         // key of tools/precompute/config.py SETTINGS
    "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0,
                  "max_tokens": 16384, "min_p": 0.05 },
    "samplingNote": "…",
    "nCtx": 20480,
    "seed": 0,
    "chatTemplate": "GGUF chat template via llama-server /apply-template, enable_thinking=true",
    "caveat": "Native recording ≠ WebGPU run token-for-token; GGUFs are not a matched quant pair."
  },
  "batchStats": {                    // over the whole batch of this setting (honest selection context)
    "n": 31, "base_cap_rate": 1.0, "edited_cap_rate": 0.5484, "base_acc": 0.0, "edited_acc": 0.3226
  },
  "defaultPrompt": {                 // how the page's default prompt did in this setting
    "qualifies": false,
    "base":   { "n_tokens": 16384, "finish": "length", "correct": false },
    "edited": { "n_tokens": 16384, "finish": "length", "correct": false }
  },
  "otherSettings": {                 // the other recorded batches, shown honestly
    "up16k_pp": { "sampling": { … }, "backend": "…", "batchStats": { … }, "defaultPrompt": { … } },
    "page4096": { … }
  },
  "settingNote": "why this setting feeds the replay (plain text)",
  "cases": [
    {
      "id": "math500-algebra-2551",
      "title": "MATH-500 algebra L1",
      "prompt": "… Please reason step by step, and put your final answer within \\boxed{}.",
      "reference": "10",
      "source": "HuggingFaceH4/MATH-500 test/algebra/24.json",
      "kind": "loop",                // "loop": base hits the cap, edited stops correct; "control": both solve
      "setting": "cap16k",           // loop cases always come from source.setting; the control may come
      "sampling": { … },             //   from another setting — then these fields name that setting
      "seed": 0, "backend": "…",
      "base": {
        "tokens": ["Here", "'s", " a", …],   // generated token pieces; "".join(tokens) is the exact text
        "logprobs": [-0.03, null, …],        // sampled-token logprob (2 decimals); null where llama-server
                                             //   merged a split UTF-8 char into the next token's entry
        "thinkEnd": null,                    // index of the "</think>" token; null = never left thinking
        "finish": "length",                  // "length" = hit max_tokens, "stop" = EOS
        "nTokens": 16384,
        "answer": null,                      // last \boxed{} after </think>, or null
        "correct": false,
        "loops": [                           // verbatim periodic repetition (tools/precompute/loops.py)
          { "start": 3052, "end": 4096, "period": 60, "repeats": 17.4 }   // [start, end) token indices
        ]
      },
      "edited": { … same shape … }
    }
  ]
}
```

Token pieces: byte-level BPE tokens that split a multi-byte UTF-8 character are merged
forward, so such a token's piece is `""` and the character appears on the next token;
the token count and indices still match the model's tokens. The prompt (chat template)
is not included in `tokens`; generation starts inside `<think>`.

`loops`: a segment is a maximal span where every token equals the token `period`
positions earlier, covering at least 3 periods and 48 tokens (periods 4–400);
non-overlapping, longest first. `repeats = (end - start) / period`.

## `tools/precompute/out/batch-summary.json` (repo only, not served)

```jsonc
{
  "source": { "engine", "llamaCpp", "host", "gguf", "samplingNote", "seed", "chatTemplate", "caveat" },
  "replaySetting": "cap16k",
  "replayCases": [ { "setting": "cap16k", "id": "…", "kind": "loop" }, … ],
  "settings": {
    "<setting>": {
      "backend": "…", "sampling": { … }, "nCtx": 20480, "seed": 0,
      "batchStats": { "n", "base_cap_rate", "edited_cap_rate", "base_acc", "edited_acc" },
      "selection": { "loop": [ids], "control": id | null },   // selection rule applied to this setting alone
      "rows": [ { "id", "level", "reference",
                  "base":   { "n_tokens", "finish", "thinkEnd", "answer", "correct", "loopTokens" },
                  "edited": { … } } ]
    }
  }
}
```

## `neuron-trace/` (index.json + one binary per case) — quantised data

Real per-token activity of the 64 edited neurons (layers 15 and 19) for every case in
`replay.json`. It comes from a **bf16** forward pass in transformers on GPU, teacher-forced over the
run's chat prompt plus its recorded tokens. Findings: `tools/precompute/out/neuron-findings.md`.

**The served values are quantised**: per-token series are int8 and heatmaps are uint8. Each
array has its own linear scale, and full-precision values exist only in the offline cache.
Load `index.json` first, then fetch one `<case-id>.bin` lazily when that case is shown
(≈ 140–290 KB each, ≈ 1.4 MB in total).

> The page reads this directory through `playground/src/neuron-trace.js` (lazy loader +
> decoder, unit-tested) and draws it in `neuron-view.js`, synced to the replay cursor. The old
> placeholder contract (`data/neuron-trace.json`) is gone.

### Runs (three per case)

| run | weights | tokens | meaning |
|---|---|---|---|
| `base` | base | `case.base.tokens` | what the original model's edited neurons did |
| `edited` | edited | `case.edited.tokens` | the edited model on its own run |
| `counterfactual` | edited | `case.base.tokens` | edited weights reading the base run's text (really run, not approximated): how hard the edit pushes back during the base loop |

### Per-token series (per run, per layer L ∈ {15, 19}), length = that run's `nTokens`

Let `a` = the input of `mlp.down_proj` (the 3584-d post-SwiGLU activation) and `y = W a` its
output, with the run's own weights. `cols` = the 64 edited columns, `u` = the edit's unit residual
direction (`weight-diff.json` `svd.u`, same sign), and `Δ` = edited − base (exact bf16).

| series | definition |
|---|---|
| `push` | `u · (Δ a)` = `Σ_k pushWeights[k] · a[cols[k]]`: the edit's added output along u. In the `base` run it is hypothetical (the edit is absent): what the edit *would* add given those activations |
| `proj` | `u · y`: the full down_proj output (all 3584 neurons) projected on u |
| `norm` | `‖a[cols]‖₂`: activity of the 64 edited neurons |

`proj(edited weights) = proj(base weights) + push` on the same inputs. A push with the opposite
sign to the base's `proj` means the edit is pushing back. The sign of u is an SVD convention.

**Token alignment.** Value `i` is taken at the position whose logits sampled generated token
`i`, so `i = 0` is the last prompt token. The forward input is `prompt + tokens[:-1]`. Index `i`
lines up with `replay.json` `case.<tokens>.tokens[i]`, `logprobs[i]` and the `loops` /
`thinkEnd` indices.

### Heatmap (per run, per layer)

`heat` = `[64, nBins]` mean `|a[cols[k]]|` per token bin, row-major (channel-major: channel `k`
is bytes `k*nBins … (k+1)*nBins-1`; channel `k` is neuron `edit[L].columns[k]`).
`nBins = min(nTokens, 256)`. Bin `b` covers tokens `[binEdges[b], binEdges[b+1])`, with
`binEdges[b] = floor(b * nTokens / nBins)`. Bins are contiguous, non-empty and cover every token.

### `index.json`

```jsonc
{
  "placeholder": false,
  "source": "bf16",
  "formatVersion": 1,
  "engine": "transformers 5.23.0 + torch 2.11.0+cu128 (CUDA 12.8, NVIDIA GeForce RTX 4090), bf16, teacher-forced, …",
  "checkpoints": { "base": { "repo", "revision" }, "edited": { "repo", "revision" } },
  "replay": "replay.json",                 // the token streams / loops / thinkEnd these traces index into
  "layers": [15, 19],
  "edit": {
    "15": { "columns": [0, 2, …],          // 64 sorted intermediate-neuron ids (= weight-diff changedColumns)
            "pushWeights": [ … ] },        // 64: u · Δ[:, col] (3 significant figures), push = pushWeights · a[cols]
    "19": { … }
  },
  "runs": { "base": { "weights": "base", "tokens": "base" },
            "edited": { "weights": "edited", "tokens": "edited" },
            "counterfactual": { "weights": "edited", "tokens": "base" } },
  "alignment": "value i = activation at the position whose logits sampled generated token i …",
  "quantized": "per-token series int8 (value = q*scale, |error| <= peak/254); heatmaps uint8 (…peak/510)",
  "cases": [
    {
      "id": "math500-algebra-2551",       // = replay.json case id (same order)
      "setting": "cap16k", "kind": "loop",
      "file": "math500-algebra-2551.bin", // relative to neuron-trace/
      "bytes": 288852,
      "promptTokens": { "base": 92, "edited": 92 },
      "runs": {
        "base": {
          "weights": "base", "tokens": "base",
          "nTokens": 16384,                 // = replay.json case.base.nTokens
          "binEdges": [0, 64, 128, …, 16384],   // nBins + 1 ints
          "forwardSeconds": 1.77,           // one RTX 4090
          "sanity": { "meanLogprobRecorded": -0.0951, "top1Agreement": 0.970, "nScored": 16384 },
          "layers": {
            "15": {
              "push": <array>, "proj": <array>, "norm": <array>,   // int8, shape [nTokens]
              "heat": <array>,                                     // uint8, shape [64, nBins]
              "channels": { "meanAbs": [ … ],   // 64: mean |a_k| over the run (3 s.f.)
                            "meanPush": [ … ] } // 64: mean pushWeights[k] · a_k (3 s.f.); sums to the run's mean push
            },
            "19": { … }
          }
        },
        "edited": { … same … },
        "counterfactual": {
          …,
          "layers": {
            // layer 15 sees exactly the base run's inputs (layers < 15 are unchanged):
            "15": { "push": { "sameAs": "base" }, "norm": { "sameAs": "base" },
                    "heat": { "sameAs": "base" }, "channels": { "sameAs": "base" },
                    "proj": <array> },          // differs: = base proj + base push
            "19": { … full … }
          }
        }
      }
    }
  ]
}
```

`<array>` = `{ "offset", "length", "dtype": "int8"|"uint8", "shape", "scale", "peak", "mean" }`.
Read `length` bytes from `offset` in the case's `.bin` as `Int8Array` / `Uint8Array`, and decode
each element as `value = q * scale`. There is no header, and arrays are packed back to back.
`scale = peak / 127` (int8, symmetric) or `peak / 255` (uint8). `peak` = max |value| and `mean` =
the full-precision mean, so the round-trip error is at most `scale / 2`, which is ≤ peak/254
(int8) or ≤ peak/510 (uint8). An all-zero array has `scale` 0.

`sanity` reports, for each recorded token, the bf16 model's log-prob of it and whether it is the
bf16 argmax at its sampling position. This shows that the bf16 forward reproduces the recording:
top-1 agreement is within 1.5 pp of the Q4_K_M recording's own rate.

`python -m tools.precompute.validate` checks the following: the schema; the case ids and
`nTokens` against `replay.json`; that the binary sizes and offsets are in range; that `binEdges`
are non-empty and cover `[0, nTokens)`; the heat shape `[64, nBins]`; that each array's int8/uint8
scale equals peak/qmax, that the peak reaches the full range, and that the dequantised mean is
within scale/2 of the stored full-precision mean (round-trip error ≤ peak/127); the
`counterfactual` layer-15 `sameAs` entries; and, per token, that
`counterfactual.proj15 ≈ base.proj15 + base.push15` within quantisation error.
