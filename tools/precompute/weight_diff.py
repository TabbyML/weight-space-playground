"""Step 2: exact bf16 weight diff, base vs edited -> playground/public/data/weight-diff.json.

    python -m tools.precompute.weight_diff [--no-gguf]

Reads both pinned bf16 safetensors directly (header + raw bytes; bf16 is widened to
f32 by a 16-bit shift, which is exact), compares every tensor byte-for-byte, and
summarises each differing tensor. Optionally (default on) dequantises the same two
tensors from each pinned GGUF with gguf-py to measure how much of the edit survives
Q4_K_M quantisation.

Conventions: HF/PyTorch weight shapes are [out, in]; ``mlp.down_proj.weight`` is
[hidden=1024, intermediate=3584] and computes ``y = W @ x`` with x the 3584-d
intermediate (post-SwiGLU) activation. A changed *column* j therefore means the
output-direction vector written by intermediate neuron j was changed. GGUF stores the
same tensor with ggml ``ne = [3584, 1024]`` (ne[0] is the contiguous/input dim), i.e.
the numpy array gguf-py returns is again [1024, 3584].
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from . import config
from .fetch import fetch_bf16, fetch_gguf

DTYPES = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I64": np.int64}
TOP_SINGULAR = 8


# ---------- pure helpers ----------

def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact bf16 -> f32 widening (bf16 is the top 16 bits of an f32)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def changed_axes(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row indices, column indices) of a 2-D delta that contain any non-zero entry."""
    nz = delta != 0
    return np.flatnonzero(nz.any(axis=1)), np.flatnonzero(nz.any(axis=0))


def rank_structure(delta_cols: np.ndarray, k: int = TOP_SINGULAR) -> dict:
    """SVD of the [out, n_changed] sub-matrix of changed columns.

    Returns top-k singular values, the energy fraction explained by the first one, the
    top left singular vector u (unit, out-dim) and the per-column coefficients c with
    rank-1 approximation delta ~= outer(u, c). The SVD sign is arbitrary; we fix it so
    that most coefficients are negative (the edit is invariant to the sign of u).
    """
    U, S, Vt = np.linalg.svd(delta_cols.astype(np.float64), full_matrices=False)
    u = U[:, 0]
    coeff = S[0] * Vt[0]
    if np.sum(coeff < 0) < np.sum(coeff > 0):
        u, coeff = -u, -coeff
    energy = S**2
    resid = delta_cols - np.outer(u, coeff)
    return {
        "singularValues": S[:k].tolist(),
        "rank1EnergyFraction": float(energy[0] / energy.sum()),
        "rank1RelResidual": float(np.linalg.norm(resid) / np.linalg.norm(delta_cols)),
        "u": u,
        "coeff": coeff,
    }


def summarize_tensor(name: str, base: np.ndarray, edited: np.ndarray) -> dict:
    delta = edited.astype(np.float64) - base.astype(np.float64)
    rows, cols = changed_axes(delta)
    sub = delta[:, cols]
    rs = rank_structure(sub)
    col_norm_base = np.linalg.norm(base[:, cols].astype(np.float64), axis=0)
    proj_before = rs["u"] @ base[:, cols].astype(np.float64)
    proj_after = rs["u"] @ edited[:, cols].astype(np.float64)
    alpha = -rs["coeff"] / proj_before
    return {
        "name": name,
        "shape": list(base.shape),
        "changedRows": int(len(rows)),
        "changedColumns": cols.tolist(),
        "deltaNorm": np.linalg.norm(sub, axis=0).tolist(),
        "baseColumnNorm": col_norm_base.tolist(),
        "maxAbsDelta": float(np.abs(delta).max()),
        "entriesChanged": int(np.count_nonzero(delta)),
        "totalEntries": int(delta.size),
        "pctChanged": 100.0 * np.count_nonzero(delta) / delta.size,
        "frobeniusDelta": float(np.linalg.norm(delta)),
        "frobeniusBase": float(np.linalg.norm(base.astype(np.float64))),
        "svd": rs,
        "projBefore": proj_before,
        "projAfter": proj_after,
        "alphaMean": float(alpha.mean()),
        "alphaStd": float(alpha.std()),
    }


def survival(bf16_delta_cols: np.ndarray, gguf_delta_cols: np.ndarray, u: np.ndarray) -> dict:
    """How much of the bf16 edit is present in a GGUF pair's dequantised delta (changed columns)."""
    a = bf16_delta_cols.ravel()
    b = gguf_delta_cols.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    proj_bf16 = u @ bf16_delta_cols
    proj_gguf = u @ gguf_delta_cols
    return {
        "cosine": cos,
        "normRatio": float(np.linalg.norm(b) / np.linalg.norm(a)),
        "projOnURatio": float(np.sum(proj_gguf * proj_bf16) / np.sum(proj_bf16 * proj_bf16)),
    }


def f16_round(x: np.ndarray, digits: int = 5) -> list[float]:
    """Round through float16 (storage precision we claim) and trim decimal noise for JSON."""
    return [float(f"{v:.{digits}g}") for v in np.asarray(x, dtype=np.float16).astype(np.float32)]


# ---------- IO ----------

def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def tensor_bytes(path: Path, header: dict, offset0: int, name: str) -> bytes:
    a, b = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(offset0 + a)
        return f.read(b - a)


def load_f32(path: Path, header: dict, offset0: int, name: str) -> np.ndarray:
    info = header[name]
    raw = np.frombuffer(tensor_bytes(path, header, offset0, name), dtype=DTYPES[info["dtype"]])
    arr = bf16_bits_to_f32(raw) if info["dtype"] == "BF16" else raw.astype(np.float32)
    return arr.reshape(info["shape"])


def hf_to_gguf_name(name: str) -> str:
    # model.language_model.layers.15.mlp.down_proj.weight -> blk.15.ffn_down.weight
    layer = name.split(".layers.")[1].split(".")[0]
    assert name.endswith("mlp.down_proj.weight"), name
    return f"blk.{layer}.ffn_down.weight"


def gguf_tensor(path: Path, name: str) -> tuple[np.ndarray, str, list[int]]:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(path)
    t = next(t for t in r.tensors if t.name == name)
    arr = dequantize(t.data, t.tensor_type).reshape(list(reversed([int(x) for x in t.shape])))
    return arr.astype(np.float32), t.tensor_type.name, [int(x) for x in t.shape]


# ---------- main ----------

def compute(with_gguf: bool = True) -> dict:
    paths = {w: fetch_bf16(w) for w in config.MODEL_KEYS}
    hdr = {w: read_safetensors_header(paths[w]) for w in config.MODEL_KEYS}
    names_b, names_e = set(hdr["base"][0]), set(hdr["edited"][0])
    assert names_b == names_e, (names_b ^ names_e)
    differing = []
    for name in sorted(names_b):
        if hdr["base"][0][name]["shape"] != hdr["edited"][0][name]["shape"]:
            raise ValueError(f"shape mismatch {name}")
        if tensor_bytes(paths["base"], *hdr["base"], name) != tensor_bytes(paths["edited"], *hdr["edited"], name):
            differing.append(name)
    print(f"{len(names_b)} tensors compared, {len(differing)} differ: {differing}")

    tensors = []
    for name in differing:
        base = load_f32(paths["base"], *hdr["base"], name)
        edited = load_f32(paths["edited"], *hdr["edited"], name)
        s = summarize_tensor(name, base, edited)
        s["ggufName"] = hf_to_gguf_name(name)
        if with_gguf:
            cols = np.array(s["changedColumns"])
            bf16_delta = (edited.astype(np.float64) - base)[:, cols]
            g = {}
            for w in config.MODEL_KEYS:
                arr, qtype, ne = gguf_tensor(fetch_gguf(w), s["ggufName"])
                g[w] = (arr, qtype, ne)
            gdelta = (g["edited"][0].astype(np.float64) - g["base"][0])
            s["gguf"] = {
                "ne": g["base"][2],
                "types": {w: g[w][1] for w in config.MODEL_KEYS},
                **survival(bf16_delta, gdelta[:, cols], s["svd"]["u"]),
                "unchangedColumnsRelDelta": float(
                    np.linalg.norm(np.delete(gdelta, cols, axis=1)) / np.linalg.norm(np.delete(base, cols, axis=1))
                ),
            }
        tensors.append(s)
    return {"compared": len(names_b), "tensors": tensors}


def to_json(result: dict) -> dict:
    out_tensors = []
    for s in result["tensors"]:
        svd = s["svd"]
        t = {
            "name": s["name"],
            "ggufName": s["ggufName"],
            "layer": int(s["name"].split(".layers.")[1].split(".")[0]),
            "shape": s["shape"],
            "changedAxis": "columns",
            "changedRows": s["changedRows"],
            "changedColumns": s["changedColumns"],
            "deltaNorm": f16_round(s["deltaNorm"]),
            "baseColumnNorm": f16_round(s["baseColumnNorm"]),
            "entriesChanged": s["entriesChanged"],
            "totalEntries": s["totalEntries"],
            "pctChanged": round(s["pctChanged"], 4),
            "maxAbsDelta": round(s["maxAbsDelta"], 6),
            "frobeniusDelta": round(s["frobeniusDelta"], 5),
            "frobeniusBase": round(s["frobeniusBase"], 5),
            "svd": {
                "singularValues": [round(v, 6) for v in svd["singularValues"]],
                "rank1EnergyFraction": round(svd["rank1EnergyFraction"], 6),
                "rank1RelResidual": round(svd["rank1RelResidual"], 6),
                "u": f16_round(svd["u"]),
                "coeff": f16_round(svd["coeff"]),
            },
            "editForm": "w_j' = w_j - alpha * (u . w_j) * u",
            "alphaMean": round(s["alphaMean"], 4),
            "alphaStd": round(s["alphaStd"], 4),
            "projBefore": f16_round(s["projBefore"]),
            "projAfter": f16_round(s["projAfter"]),
        }
        if "gguf" in s:
            g = s["gguf"]
            t["gguf"] = {
                "ne": g["ne"],
                "types": g["types"],
                "cosine": round(g["cosine"], 5),
                "normRatio": round(g["normRatio"], 5),
                "projOnURatio": round(g["projOnURatio"], 5),
                "unchangedColumnsRelDelta": round(g["unchangedColumnsRelDelta"], 5),
            }
        out_tensors.append(t)
    return {
        "placeholder": False,
        "source": "bf16 safetensors",
        "quant": "bf16",
        "base": {"repo": config.BF16["base"]["repo"], "revision": config.BF16["base"]["revision"]},
        "edited": {"repo": config.BF16["edited"]["repo"], "revision": config.BF16["edited"]["revision"]},
        "ggufPair": {w: config.GGUF[w]["url"] for w in config.MODEL_KEYS},
        "convention": (
            "HF [out, in] = [hidden 1024, intermediate 3584]; changedColumns are down_proj INPUT "
            "columns = intermediate (MLP) neurons; u lives in the 1024-d residual stream; "
            "delta[:, j] ~= u * coeff[k] for j = changedColumns[k]; coeff[k] = -alpha * projBefore[k] "
            "with projBefore[k] = u . base[:, j]"
        ),
        "tensorsCompared": result["compared"],
        "tensors": out_tensors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gguf", action="store_true")
    a = ap.parse_args()
    res = compute(with_gguf=not a.no_gguf)
    data = to_json(res)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.DATA_DIR / "weight-diff.json"
    dst.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    for t in data["tensors"]:
        sv = t["svd"]["singularValues"]
        print(f"{t['name']}: shape {t['shape']} rows changed {t['changedRows']} cols {len(t['changedColumns'])} "
              f"entries {t['entriesChanged']} max|d| {t['maxAbsDelta']} sv {sv[:4]} "
              f"rank1 energy {t['svd']['rank1EnergyFraction']} gguf {t.get('gguf')}")
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
