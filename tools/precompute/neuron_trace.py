"""Step 6: per-token activity of the 64 edited neurons -> playground/public/data/neuron-trace/.

    python -m tools.precompute.neuron_trace [--gpu 0] [--force] [--cases ID ...]

For every case in replay.json, a **bf16** teacher-forced forward pass (transformers, GPU)
over the run's own chat prompt + its recorded tokens, three runs per case:

    base            base weights   over the base run's tokens
    edited          edited weights over the edited run's tokens
    counterfactual  edited weights over the base run's tokens (what the edit would have done
                    during the base loop; actually run, not approximated)

The input ``a`` of ``mlp.down_proj`` (the 3584-d post-SwiGLU intermediate activation) and its
output ``y`` are hooked at layers 15 and 19. Per generated token we keep, per layer:

    push   = u . (Delta @ a)   the edit's added output along its residual direction u
                               (= sum_k pushWeight[k] * a[col_k]; Delta = edited - base, exact bf16)
    proj   = u . y             the full down_proj output (all 3584 neurons, the run's own weights)
    norm   = ||a[cols]||_2     activity of the 64 edited neurons

plus a binned heatmap of mean |a[col_k]| (64 channels x <= 256 token bins).

Token alignment: value ``i`` is taken at the sequence position whose logits sampled generated
token ``i`` (the last prompt token for i = 0), i.e. the forward input is prompt + tokens[:-1].
Prompt ids come from llama.cpp's own tokenizer (llama-tokenize on the pinned GGUF, same as
llama-server) and are asserted equal to the HF tokenizer's ids and to ``tokens_evaluated``;
generated ids are the recorded ids, asserted to decode (HF tokenizer) to exactly the recorded
text and to name the same vocabulary entries as the GGUF vocab.

Full-precision results are cached per run under CACHE/neuron-trace/ (``.npz`` + ``.json``);
the served files are int8/uint8-quantised (see playground/public/data/README.md).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config
from .fetch import fetch_bf16
from .weight_diff import load_f32, rank_structure, read_safetensors_header

LAYERS = (15, 19)
RUNS = ("base", "edited", "counterfactual")
# run -> (weights, token stream)
RUN_SPEC = {"base": ("base", "base"), "edited": ("edited", "edited"), "counterfactual": ("edited", "base")}
MAX_BINS = 256
TRACE_DIR = config.CACHE / "neuron-trace"
DATA_SUBDIR = config.DATA_DIR / "neuron-trace"
LLAMA_TOKENIZE = config.CACHE / "llama-vulkan" / "llama-b12211" / "llama-tokenize"
SERIES = ("push", "proj", "norm")
FORMAT_VERSION = 1


def tensor_name(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.down_proj.weight"


# ---------- pure helpers ----------

def bin_edges(n: int, max_bins: int = MAX_BINS) -> list[int]:
    """Edges of min(n, max_bins) contiguous, non-empty token bins covering [0, n)."""
    if n <= 0:
        raise ValueError("need at least one token")
    nb = min(n, max_bins)
    return [(b * n) // nb for b in range(nb + 1)]


def binned_mean_abs(a: np.ndarray, edges: list[int]) -> np.ndarray:
    """[T, C] activations -> [C, n_bins] mean |a| per bin."""
    absa = np.abs(np.asarray(a, dtype=np.float64))
    return np.stack([absa[s:e].mean(axis=0) for s, e in zip(edges[:-1], edges[1:])], axis=1)


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8: q = round(x / scale), scale = max|x| / 127 (scale 0 for an all-zero series)."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        return np.zeros(x.shape, dtype=np.int8), 0.0
    scale = peak / 127.0
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), scale


def quantize_uint8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Non-negative uint8: q = round(x / scale), scale = max(x) / 255."""
    x = np.asarray(x, dtype=np.float64)
    if x.size and x.min() < 0:
        raise ValueError("quantize_uint8 needs non-negative values")
    peak = float(x.max()) if x.size else 0.0
    if peak == 0.0:
        return np.zeros(x.shape, dtype=np.uint8), 0.0
    scale = peak / 255.0
    return np.clip(np.rint(x / scale), 0, 255).astype(np.uint8), scale


def sig(x: float, digits: int = 3) -> float:
    """Round to `digits` significant figures (JSON-friendly float)."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return 0.0 if x == 0.0 else x
    return float(f"{x:.{digits - 1}e}")


def push_weights(u: np.ndarray, delta_cols: np.ndarray) -> np.ndarray:
    """Per changed column k: u . Delta[:, col_k], so push = pushWeights . a[cols]."""
    return np.asarray(u, dtype=np.float64) @ np.asarray(delta_cols, dtype=np.float64)


def parse_llama_tokenize_ids(stdout: str) -> list[int]:
    """Last ``[id, id, ...]`` line printed by ``llama-tokenize --ids``."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return [int(t) for t in json.loads(line)]
    raise ValueError("no id list in llama-tokenize output")


def forward_ids(prompt_ids: list[int], gen_ids: list[int]) -> tuple[list[int], int]:
    """Teacher-forced input (prompt + gen[:-1]) and the position whose logits sampled gen[0]."""
    if not prompt_ids or not gen_ids:
        raise ValueError("need a prompt and at least one generated token")
    return list(prompt_ids) + list(gen_ids[:-1]), len(prompt_ids) - 1


class Packer:
    """Appends quantised arrays to one binary blob and returns index entries for them."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.size = 0

    def add(self, q: np.ndarray, scale: float, x: np.ndarray, shape: list[int]) -> dict:
        x = np.asarray(x, dtype=np.float64)
        entry = {
            "offset": self.size,
            "length": int(q.size),
            "dtype": "int8" if q.dtype == np.int8 else "uint8",
            "shape": shape,
            "scale": scale,                         # value = q * scale (full float precision)
            "peak": float(np.max(np.abs(x))) if x.size else 0.0,
            "mean": float(x.mean()) if x.size else 0.0,
        }
        b = q.tobytes()
        self.parts.append(b)
        self.size += len(b)
        return entry

    def series(self, x: np.ndarray) -> dict:
        q, s = quantize_int8(x)
        assert np.max(np.abs(q * s - np.asarray(x, dtype=np.float64)), initial=0.0) <= s / 2 + 1e-12
        return self.add(q, s, x, [int(q.size)])

    def heat(self, h: np.ndarray) -> dict:
        q, s = quantize_uint8(h)
        assert np.max(np.abs(q * s - h), initial=0.0) <= s / 2 + 1e-12
        return self.add(q.reshape(-1), s, h, list(h.shape))

    def blob(self) -> bytes:
        return b"".join(self.parts)


def channel_summary(a: np.ndarray, w: np.ndarray) -> dict:
    """Per channel: mean |a| and mean signed push contribution w_k * a_k over the run."""
    a = np.asarray(a, dtype=np.float64)
    return {
        "meanAbs": [sig(v) for v in np.abs(a).mean(axis=0)],
        "meanPush": [sig(v) for v in (a * w).mean(axis=0)],
    }


def run_record(arrs: dict, w: dict) -> dict:
    """Derived per-token series from the cached raw arrays of one run (all float64)."""
    out = {}
    for L in LAYERS:
        a = arrs[f"a{L}"].astype(np.float64)
        out[L] = {
            "a": a,
            "push": a @ w[L],
            "proj": arrs[f"proj{L}"].astype(np.float64),
            "norm": np.linalg.norm(a, axis=1),
        }
    return out


# ---------- inputs ----------

def load_replay() -> dict:
    return json.loads((config.DATA_DIR / "replay.json").read_text())


def raw_recording(case: dict, model: str) -> dict:
    return json.loads((config.RAW_DIR / case["setting"] / model / f"{case['id']}__s0.json").read_text())


def llama_prompt_ids(rendered_prompt: str, model: str) -> list[int]:
    tmp = TRACE_DIR / "prompt.txt"
    tmp.write_text(rendered_prompt)
    r = subprocess.run(
        [str(LLAMA_TOKENIZE), "-m", str(config.gguf_path(model)), "-f", str(tmp), "--ids", "--log-disable"],
        capture_output=True, text=True, check=True, timeout=300,
    )
    return parse_llama_tokenize_ids(r.stdout)


def edit_tensors() -> dict:
    """Per layer: changed columns, exact bf16 delta columns, u, push weights, edited weight."""
    wd = json.loads((config.DATA_DIR / "weight-diff.json").read_text())
    names = sorted(t["name"] for t in wd["tensors"])
    assert names == sorted(tensor_name(L) for L in LAYERS), f"unexpected differing tensors {names}"
    paths = {w: fetch_bf16(w) for w in config.MODEL_KEYS}
    hdr = {w: read_safetensors_header(paths[w]) for w in config.MODEL_KEYS}
    out = {}
    for t in wd["tensors"]:
        L = t["layer"]
        base = load_f32(paths["base"], *hdr["base"], t["name"])
        edited = load_f32(paths["edited"], *hdr["edited"], t["name"])
        delta = edited.astype(np.float64) - base.astype(np.float64)
        cols = np.array(t["changedColumns"])
        assert np.all(delta[:, np.setdiff1d(np.arange(delta.shape[1]), cols)] == 0)
        rs = rank_structure(delta[:, cols])
        u = rs["u"]
        cos = float(u @ np.array(t["svd"]["u"]) / np.linalg.norm(t["svd"]["u"]))
        assert cos > 0.999, f"u does not match weight-diff.json (cos {cos})"
        out[L] = {"cols": cols, "u": u, "w": push_weights(u, delta[:, cols]), "edited": edited}
    return out


# ---------- GPU forward ----------

def versions() -> dict:
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def load_model(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = fetch_bf16("base").parent
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return tok, model


def set_weights(model, edits: dict, which: str, base_w: dict) -> None:
    import torch

    for L in LAYERS:
        W = model.model.layers[L].mlp.down_proj.weight
        src = base_w[L] if which == "base" else torch.from_numpy(edits[L]["edited"]).to(W.dtype)
        with torch.no_grad():
            W.copy_(src.to(W.device))


def forward_run(model, ids: list[int], start: int, gen: list[int], edits: dict) -> tuple[dict, dict]:
    """Hooked teacher-forced forward; returns raw arrays [n_gen, ...] (float32) and sanity stats."""
    import torch

    dev = next(model.parameters()).device
    cap: dict[str, np.ndarray] = {}
    hooks = []
    for L in LAYERS:
        cols = torch.tensor(edits[L]["cols"], device=dev)
        u = torch.tensor(edits[L]["u"], device=dev, dtype=torch.float32)

        def hook(mod, inp, out, L=L, cols=cols, u=u):
            a = inp[0][0, start:].float()
            cap[f"a{L}"] = a[:, cols].cpu().numpy()
            cap[f"proj{L}"] = (out[0, start:].float() @ u).cpu().numpy()

        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_hook(hook))
    x = torch.tensor([ids], device=dev)
    try:
        with torch.no_grad():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, start:]
            # sanity: bf16 log-prob / top-1 of every recorded token at its sampling position (chunked lm_head)
            nxt = torch.tensor(gen, device=dev)
            lp, top1 = [], []
            for s in range(0, h.shape[0], 2048):
                logits = model.lm_head(h[s:s + 2048]).float().log_softmax(-1)
                tgt = nxt[s:s + 2048]
                k = tgt.shape[0]
                lp.append(logits[:k].gather(1, tgt[:, None])[:, 0].cpu())
                top1.append((logits[:k].argmax(-1) == tgt).cpu())
    finally:
        for hk in hooks:
            hk.remove()
    lp_t, top1_t = torch.cat(lp), torch.cat(top1)
    cap["logprob"] = lp_t.numpy()  # bf16 log-prob of each recorded token (cache only; used by the findings)
    stats = {"meanLogprobRecorded": float(lp_t.mean()), "top1Agreement": float(top1_t.float().mean()),
             "nScored": int(lp_t.numel())}
    return cap, stats


def check_tokens(tok, case: dict, model_key: str, raw: dict, vocab: dict) -> tuple[list[int], list[int], dict]:
    """Prompt + generated ids for one recorded stream, with the exactness checks."""
    resp = raw["response"]
    gen = list(resp["tokens"])
    assert len(gen) == case[model_key]["nTokens"], "recorded ids != replay nTokens"
    llama_ids = llama_prompt_ids(raw["rendered_prompt"], model_key)
    hf_ids = tok(raw["rendered_prompt"], add_special_tokens=False)["input_ids"]
    assert llama_ids == hf_ids, "prompt tokenisation differs between llama.cpp and HF"
    assert len(llama_ids) == resp["tokens_evaluated"], "prompt length != tokens_evaluated"
    hf_tokens = tok.convert_ids_to_tokens(gen)
    assert all(hf_tokens[i] == vocab["tokens"][t] for i, t in enumerate(gen)), "vocab mismatch HF vs GGUF"
    # llama-server leaves the final EOS (a control token) out of `content`
    text_ids = gen[:-1] if gen[-1] in tok.all_special_ids else gen
    text = tok.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert text == resp["content"], "HF decode of recorded ids != recorded text"
    # informational only (we feed the recorded ids): does re-encoding the text give the same ids?
    reenc = tok(resp["content"], add_special_tokens=False)["input_ids"]
    first_div = next((i for i, (x, y) in enumerate(zip(reenc, text_ids)) if x != y), None)
    if first_div is None and len(reenc) != len(text_ids):
        first_div = min(len(reenc), len(text_ids))
    info = {
        "promptTokens": len(llama_ids),
        "promptIdsMatch": "llama-tokenize == HF tokenizer == tokens_evaluated",
        "genIdsDecodeExact": True,
        "reencodeEqualsRecorded": first_div is None,
        "reencodeFirstDivergence": first_div,
    }
    return llama_ids, gen, info


def compute(device: str, force: bool, only: list[str] | None) -> None:
    import torch

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    replay = load_replay()
    cases = [c for c in replay["cases"] if not only or c["id"] in only]
    todo = [(c, r) for c in cases for r in RUNS
            if force or not (TRACE_DIR / c["setting"] / f"{c['id']}__{r}.npz").exists()]
    if not todo:
        print("neuron-trace: all runs cached")
        return
    edits = edit_tensors()
    tok, model = load_model(device)
    base_w = {L: model.model.layers[L].mlp.down_proj.weight.detach().clone() for L in LAYERS}
    vocab = {m: json.loads((config.CACHE / f"vocab-{m}.json").read_text()) for m in config.MODEL_KEYS}
    vers = versions()
    current = None
    for c, run in todo:
        weights, stream = RUN_SPEC[run]
        if current != weights:
            set_weights(model, edits, weights, base_w)
            current = weights
        raw = raw_recording(c, stream)
        prompt_ids, gen, tok_info = check_tokens(tok, c, stream, raw, vocab[stream])
        ids, start = forward_ids(prompt_ids, gen)
        torch.cuda.synchronize()
        t0 = time.time()
        cap, sanity = forward_run(model, ids, start, gen, edits)
        torch.cuda.synchronize()
        dt = time.time() - t0
        assert all(cap[f"a{L}"].shape == (len(gen), 64) for L in LAYERS)
        out = TRACE_DIR / c["setting"] / f"{c['id']}__{run}"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".npz"), **cap)
        meta = {"case": c["id"], "setting": c["setting"], "run": run, "weights": weights, "tokens": stream,
                "nTokens": len(gen), "seqLen": len(ids), "forwardSeconds": round(dt, 3), "versions": vers,
                "tokenisation": tok_info, "sanity": sanity, "device": device}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"{c['id']:32s} {run:15s} n={len(gen):6d} {dt:6.2f}s top1={sanity['top1Agreement']:.3f} "
              f"lp={sanity['meanLogprobRecorded']:.3f}")


# ---------- served files ----------

def load_cached(case: dict, run: str) -> tuple[dict, dict]:
    p = TRACE_DIR / case["setting"] / f"{case['id']}__{run}"
    arrs = dict(np.load(p.with_suffix(".npz")))
    return arrs, json.loads(p.with_suffix(".json").read_text())


def build_files() -> dict:
    replay = load_replay()
    edits = edit_tensors()
    w = {L: edits[L]["w"] for L in LAYERS}
    DATA_SUBDIR.mkdir(parents=True, exist_ok=True)
    index_cases = []
    meta0 = None
    for c in replay["cases"]:
        recs, metas = {}, {}
        for run in RUNS:
            arrs, meta = load_cached(c, run)
            recs[run], metas[run] = run_record(arrs, w), meta
            meta0 = meta0 or meta
        # counterfactual layer 15 sees exactly the base run's inputs (layers < 15 are unchanged)
        assert np.array_equal(recs["counterfactual"][15]["a"], recs["base"][15]["a"]), "cf L15 a != base L15 a"
        pk = Packer()
        runs_idx = {}
        for run in RUNS:
            weights, stream = RUN_SPEC[run]
            n = c[stream]["nTokens"]
            edges = bin_edges(n)
            layers_idx = {}
            for L in LAYERS:
                r = recs[run][L]
                assert len(r["push"]) == n
                if run == "counterfactual" and L == 15:
                    layers_idx[str(L)] = {"push": {"sameAs": "base"}, "norm": {"sameAs": "base"},
                                          "proj": pk.series(r["proj"]), "heat": {"sameAs": "base"},
                                          "channels": {"sameAs": "base"}}
                    continue
                layers_idx[str(L)] = {
                    "push": pk.series(r["push"]),
                    "proj": pk.series(r["proj"]),
                    "norm": pk.series(r["norm"]),
                    "heat": pk.heat(binned_mean_abs(r["a"], edges)),
                    "channels": channel_summary(r["a"], w[L]),
                }
            runs_idx[run] = {"weights": weights, "tokens": stream, "nTokens": n, "binEdges": edges,
                             "forwardSeconds": metas[run]["forwardSeconds"],
                             "sanity": {k: sig(v, 4) if isinstance(v, float) else v
                                        for k, v in metas[run]["sanity"].items()},
                             "layers": layers_idx}
        blob = pk.blob()
        fname = f"{c['id']}.bin"
        (DATA_SUBDIR / fname).write_bytes(blob)
        index_cases.append({"id": c["id"], "setting": c["setting"], "kind": c["kind"], "file": fname,
                            "bytes": len(blob), "promptTokens": {s: metas[r]["tokenisation"]["promptTokens"]
                                                                  for r, s in (("base", "base"), ("edited", "edited"))},
                            "runs": runs_idx})
    index = {
        "placeholder": False,
        "source": "bf16",
        "formatVersion": FORMAT_VERSION,
        "engine": f"transformers {meta0['versions']['transformers']} + torch {meta0['versions']['torch']} "
                  f"(CUDA {meta0['versions']['cuda']}, {meta0['versions']['gpu']}), bf16, teacher-forced, "
                  "reference PyTorch kernels for the Gated DeltaNet layers (no fla / causal_conv1d)",
        "checkpoints": config.BF16,
        "replay": "replay.json",
        "layers": list(LAYERS),
        "edit": {str(L): {"columns": edits[L]["cols"].tolist(), "pushWeights": [sig(v) for v in w[L]]}
                 for L in LAYERS},
        "runs": {r: {"weights": RUN_SPEC[r][0], "tokens": RUN_SPEC[r][1]} for r in RUNS},
        "alignment": "value i = activation at the position whose logits sampled generated token i "
                     "(forward input = prompt + tokens[:-1]; i=0 is the last prompt token)",
        "quantized": "per-token series int8 (value = q*scale, |error| <= peak/254); heatmaps uint8 "
                     "(value = q*scale, |error| <= peak/510)",
        "cases": index_cases,
    }
    (DATA_SUBDIR / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute cached runs")
    ap.add_argument("--cases", nargs="*", help="only these case ids (default: all replay cases)")
    ap.add_argument("--no-compute", action="store_true", help="only rebuild the served files from the cache")
    a = ap.parse_args()
    if not a.no_compute:
        compute(f"cuda:{a.gpu}", a.force, a.cases)
    idx = build_files()
    total = sum(p.stat().st_size for p in DATA_SUBDIR.iterdir())
    print(f"wrote {DATA_SUBDIR} ({len(idx['cases'])} cases, {total:,} bytes)")


if __name__ == "__main__":
    main()
