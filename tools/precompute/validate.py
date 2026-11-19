"""Schema + consistency check for the precomputed data files (no third-party deps).

    python -m tools.precompute.validate [files...]

Defaults to playground/public/data/{weight-diff,replay}.json and
tools/precompute/out/batch-summary.json (and data/neuron-trace/index.json + its binaries if present).
Exits non-zero with a list of problems if anything is off.
"""
from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path

from . import config


class Checker:
    def __init__(self, fname: str):
        self.fname = fname
        self.errors: list[str] = []

    def req(self, obj, key: str, typ, path: str):
        if not isinstance(obj, dict) or key not in obj:
            self.errors.append(f"{self.fname}: missing {path}.{key}")
            return None
        v = obj[key]
        if typ is not None and not isinstance(v, typ):
            self.errors.append(f"{self.fname}: {path}.{key} should be {typ}, got {type(v).__name__}")
            return None
        return v

    def check(self, cond: bool, msg: str):
        if not cond:
            self.errors.append(f"{self.fname}: {msg}")


NUM = (int, float)


def check_weight_diff(d: dict, c: Checker) -> None:
    c.check(d.get("placeholder") is False, "placeholder must be false")
    c.req(d, "source", str, "")
    c.req(d, "convention", str, "")
    tensors = c.req(d, "tensors", list, "") or []
    c.check(len(tensors) > 0, "no tensors")
    for i, t in enumerate(tensors):
        p = f"tensors[{i}]"
        for k, typ in (("name", str), ("ggufName", str), ("layer", int), ("shape", list), ("changedAxis", str),
                       ("changedColumns", list), ("deltaNorm", list), ("baseColumnNorm", list),
                       ("entriesChanged", int), ("totalEntries", int), ("pctChanged", NUM), ("maxAbsDelta", NUM),
                       ("svd", dict), ("alphaMean", NUM), ("projBefore", list), ("projAfter", list)):
            c.req(t, k, typ, p)
        cols = t.get("changedColumns") or []
        shape = t.get("shape") or [0, 0]
        n = len(cols)
        c.check(len(shape) == 2, f"{p}.shape must be 2-D")
        c.check(all(isinstance(j, int) and 0 <= j < shape[1] for j in cols), f"{p}.changedColumns out of range")
        c.check(cols == sorted(set(cols)), f"{p}.changedColumns must be sorted unique")
        for k in ("deltaNorm", "baseColumnNorm", "projBefore", "projAfter"):
            c.check(len(t.get(k) or []) == n, f"{p}.{k} length != changedColumns")
        svd = t.get("svd") or {}
        c.check(len(svd.get("u") or []) == shape[0], f"{p}.svd.u length != out dim")
        c.check(len(svd.get("coeff") or []) == n, f"{p}.svd.coeff length != changedColumns")
        sv = svd.get("singularValues") or []
        c.check(sv == sorted(sv, reverse=True), f"{p}.svd.singularValues not descending")
        c.check(0 <= t.get("entriesChanged", -1) <= t.get("totalEntries", 0), f"{p}.entriesChanged out of range")


def check_run(r: dict, c: Checker, p: str) -> None:
    toks = c.req(r, "tokens", list, p) or []
    c.check(all(isinstance(x, str) for x in toks), f"{p}.tokens must be strings")
    lps = c.req(r, "logprobs", list, p)
    if lps is not None:
        c.check(len(lps) == len(toks), f"{p}.logprobs length != tokens")
        c.check(all(x is None or isinstance(x, NUM) for x in lps), f"{p}.logprobs must be numbers or null")
    c.check(r.get("nTokens") == len(toks), f"{p}.nTokens != len(tokens)")
    c.check(r.get("finish") in ("stop", "length"), f"{p}.finish must be stop|length")
    te = r.get("thinkEnd", "missing")
    c.check(te is None or (isinstance(te, int) and 0 <= te < len(toks) and toks[te] == "</think>"),
            f"{p}.thinkEnd must be null or index of '</think>'")
    c.check(isinstance(r.get("correct"), bool), f"{p}.correct must be bool")
    c.check(r.get("answer") is None or isinstance(r.get("answer"), str), f"{p}.answer must be str|null")
    prev_end = 0
    for j, s in enumerate(c.req(r, "loops", list, p) or []):
        ok = (isinstance(s, dict) and all(isinstance(s.get(k), int) for k in ("start", "end", "period"))
              and isinstance(s.get("repeats"), NUM) and prev_end <= s["start"] < s["end"] <= len(toks)
              and s["period"] > 0)
        c.check(ok, f"{p}.loops[{j}] invalid or overlapping")
        if ok:
            prev_end = s["end"]


def check_stats(s: dict, c: Checker, p: str) -> None:
    c.req(s, "n", int, p)
    for k in ("base_cap_rate", "edited_cap_rate", "base_acc", "edited_acc"):
        v = c.req(s, k, NUM, p)
        c.check(v is None or 0 <= v <= 1, f"{p}.{k} must be in [0,1]")


def check_replay(d: dict, c: Checker) -> None:
    c.check(d.get("placeholder") is False and d.get("recorded") is True, "placeholder=false, recorded=true required")
    src = c.req(d, "source", dict, "") or {}
    for k in ("engine", "llamaCpp", "gguf", "sampling", "seed"):
        c.req(src, k, None, "source")
    check_stats(c.req(d, "batchStats", dict, "") or {}, c, "batchStats")
    for name, o in (c.req(d, "otherSettings", dict, "") or {}).items():
        c.check(name in config.SETTINGS and name != src.get("setting"), f"otherSettings.{name}: bad setting")
        check_stats(c.req(o, "batchStats", dict, f"otherSettings.{name}") or {}, c, f"otherSettings.{name}.batchStats")
    cases = c.req(d, "cases", list, "") or []
    ids = [x.get("id") for x in cases]
    c.check(len(ids) == len(set(ids)), "duplicate case ids")
    for i, x in enumerate(cases):
        p = f"cases[{i}]"
        for k in ("id", "title", "prompt", "reference", "setting", "backend"):
            c.req(x, k, str, p)
        c.check(x.get("setting") in config.SETTINGS, f"{p}.setting unknown")
        c.req(x, "sampling", dict, p)
        c.check(x.get("kind") in ("loop", "control"), f"{p}.kind must be loop|control")
        for w in config.MODEL_KEYS:
            check_run(c.req(x, w, dict, p) or {}, c, f"{p}.{w}")
        if x.get("kind") == "loop":
            c.check(x.get("setting") == src.get("setting"), f"{p}: loop cases must come from source.setting")
            c.check(x.get("base", {}).get("finish") == "length" and x.get("edited", {}).get("finish") == "stop",
                    f"{p}: loop case must have base=length, edited=stop")


def check_batch_summary(d: dict, c: Checker) -> None:
    settings = c.req(d, "settings", dict, "") or {}
    c.check(d.get("replaySetting") in settings, "replaySetting must name a recorded setting")
    for name, st in settings.items():
        p = f"settings.{name}"
        c.check(name in config.SETTINGS, f"{p}: unknown setting")
        c.req(st, "sampling", dict, p)
        c.req(st, "backend", str, p)
        check_stats(c.req(st, "batchStats", dict, p) or {}, c, f"{p}.batchStats")
        rows = c.req(st, "rows", list, p) or []
        c.check(len(rows) == st.get("batchStats", {}).get("n"), f"{p}: batchStats.n != len(rows)")
        for i, r in enumerate(rows):
            for w in config.MODEL_KEYS:
                m = c.req(r, w, dict, f"{p}.rows[{i}]") or {}
                c.req(m, "n_tokens", int, f"{p}.rows[{i}].{w}")
                c.check(m.get("finish") in ("stop", "length"), f"{p}.rows[{i}].{w}.finish")
                c.req(m, "correct", bool, f"{p}.rows[{i}].{w}")


NT_SERIES = ("push", "proj", "norm")
NT_RUNS = {"base": ("base", "base"), "edited": ("edited", "edited"), "counterfactual": ("edited", "base")}
NT_SAME_AS_BASE = ("push", "norm", "heat", "channels")  # counterfactual layer 15 == base layer 15


def check_quantised(blob: bytes, e: dict, shape: list[int], dtype: str, c: Checker, p: str) -> list[float] | None:
    """Layout + round-trip checks of one quantised array; returns its dequantised values."""
    ok = (isinstance(e, dict) and e.get("dtype") == dtype and e.get("shape") == shape
          and all(isinstance(e.get(k), NUM) for k in ("offset", "length", "scale", "peak", "mean")))
    c.check(ok, f"{p}: bad entry (want {dtype} {shape})")
    if not ok:
        return None
    n = 1
    for k in shape:
        n *= k
    c.check(e["length"] == n and 0 <= e["offset"] and e["offset"] + n <= len(blob), f"{p}: length/offset out of range")
    if e["length"] != n or e["offset"] + n > len(blob):
        return None
    qmax = 127 if dtype == "int8" else 255
    q = array("b" if dtype == "int8" else "B", blob[e["offset"]:e["offset"] + n])
    peak, scale = e["peak"], e["scale"]
    if peak == 0:
        c.check(scale == 0 and not any(q), f"{p}: zero series must have scale 0 and q 0")
    else:
        # scale = peak / qmax and the peak hits full range, so every value's rounding error is
        # <= scale / 2 = peak / (2 * qmax) <= peak / 127
        c.check(abs(scale * qmax - peak) <= 1e-9 * peak, f"{p}: scale != peak / {qmax}")
        c.check(max(abs(v) for v in q) == qmax, f"{p}: peak does not reach the full quantised range")
        mean_err = abs(sum(q) * scale / n - e["mean"])
        c.check(mean_err <= scale / 2 + 1e-9 * peak, f"{p}: round-trip mean error {mean_err:.3g} > scale/2")
        c.check(scale / 2 <= peak / 127, f"{p}: round-trip error bound exceeds peak/127")
    return [v * scale for v in q]


def check_neuron_trace_index(d: dict, c: Checker, path: Path) -> None:
    c.check(d.get("placeholder") is False, "placeholder must be false")
    c.check(d.get("source") == "bf16", "source must be 'bf16'")
    c.check(d.get("formatVersion") == 1, "formatVersion must be 1")
    for k in ("engine", "alignment", "quantized"):
        c.req(d, k, str, "")
    layers = c.req(d, "layers", list, "") or []
    c.check(layers == [15, 19], "layers must be [15, 19]")
    edit = c.req(d, "edit", dict, "") or {}
    for L in layers:
        e = edit.get(str(L)) or {}
        cols = e.get("columns") or []
        c.check(len(cols) == 64 and cols == sorted(set(cols)), f"edit.{L}.columns must be 64 sorted unique ids")
        c.check(len(e.get("pushWeights") or []) == 64, f"edit.{L}.pushWeights must have 64 entries")
    c.check({r: [v.get("weights"), v.get("tokens")] for r, v in (d.get("runs") or {}).items()}
            == {r: list(v) for r, v in NT_RUNS.items()}, "runs must be base/edited/counterfactual with their spec")
    replay_path = path.parent.parent / "replay.json"
    replay = {x["id"]: x for x in json.loads(replay_path.read_text())["cases"]} if replay_path.exists() else None
    c.check(replay is not None, "replay.json not found next to neuron-trace/")
    cases = c.req(d, "cases", list, "") or []
    c.check(replay is None or sorted(x.get("id") for x in cases) == sorted(replay), "case ids != replay.json case ids")
    for i, x in enumerate(cases):
        p = f"cases[{i}]"
        f = path.parent / str(x.get("file"))
        if not f.is_file():
            c.check(False, f"{p}: binary {f.name} missing")
            continue
        blob = f.read_bytes()
        c.check(len(blob) == x.get("bytes"), f"{p}: {f.name} size != bytes")
        rc = (replay or {}).get(x.get("id")) or {}
        runs = c.req(x, "runs", dict, p) or {}
        c.check(sorted(runs) == sorted(NT_RUNS), f"{p}: runs must be {sorted(NT_RUNS)}")
        vals: dict = {}
        for run, r in runs.items():
            rp = f"{p}.runs.{run}"
            n = r.get("nTokens")
            c.check(isinstance(n, int) and n > 0, f"{rp}.nTokens must be a positive int")
            if not isinstance(n, int) or n <= 0:
                continue
            if run in NT_RUNS:
                c.check([r.get("weights"), r.get("tokens")] == list(NT_RUNS[run]), f"{rp}: weights/tokens spec")
                c.check(rc.get(NT_RUNS[run][1], {}).get("nTokens") == n, f"{rp}.nTokens != replay.json token count")
            edges = r.get("binEdges") or []
            nb = len(edges) - 1
            c.check(1 <= nb <= 256 and edges[0] == 0 and edges[-1] == n
                    and all(a < b for a, b in zip(edges, edges[1:])),
                    f"{rp}.binEdges must be 1..256 non-empty bins covering [0, nTokens)")
            for L in layers:
                lp = f"{rp}.layers.{L}"
                lay = (r.get("layers") or {}).get(str(L)) or {}
                same = run == "counterfactual" and L == 15
                for k in NT_SERIES:
                    if same and k in NT_SAME_AS_BASE:
                        c.check(lay.get(k) == {"sameAs": "base"}, f"{lp}.{k} must be {{sameAs: base}}")
                        continue
                    vals[(run, L, k)] = (check_quantised(blob, lay.get(k), [n], "int8", c, f"{lp}.{k}"), lay.get(k))
                if same:
                    for k in ("heat", "channels"):
                        c.check(lay.get(k) == {"sameAs": "base"}, f"{lp}.{k} must be {{sameAs: base}}")
                    continue
                check_quantised(blob, lay.get("heat"), [64, max(nb, 0)], "uint8", c, f"{lp}.heat")
                ch = lay.get("channels") or {}
                c.check(len(ch.get("meanAbs") or []) == 64 and len(ch.get("meanPush") or []) == 64,
                        f"{lp}.channels.meanAbs/meanPush must have 64 entries")
        # consistency: counterfactual L15 proj = base L15 proj + base L15 push (same inputs, W + Delta)
        trio = [vals.get(k) for k in (("counterfactual", 15, "proj"), ("base", 15, "proj"), ("base", 15, "push"))]
        if all(t and t[0] for t in trio):
            tol = sum(t[1]["scale"] for t in trio) / 2 + 0.005
            err = max(abs(a - b - q) for a, b, q in zip(trio[0][0], trio[1][0], trio[2][0]))
            c.check(err <= tol, f"{p}: counterfactual L15 proj != base proj + push (max err {err:.3g} > {tol:.3g})")


CHECKS = {
    "weight-diff.json": check_weight_diff,
    "replay.json": check_replay,
    "batch-summary.json": check_batch_summary,
}
NEURON_TRACE_INDEX = config.DATA_DIR / "neuron-trace" / "index.json"


def validate(path: Path) -> list[str]:
    c = Checker(path.name)
    try:
        d = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return [f"{path.name}: unreadable JSON: {e}"]
    if path.name == "index.json" and path.parent.name == "neuron-trace":
        check_neuron_trace_index(d, c, path)
    else:
        CHECKS[path.name](d, c)
    return c.errors


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [
        config.DATA_DIR / "weight-diff.json",
        config.DATA_DIR / "replay.json",
        config.OUT_DIR / "batch-summary.json",
    ] + ([NEURON_TRACE_INDEX] if NEURON_TRACE_INDEX.exists() else [])
    errors: list[str] = []
    for p in paths:
        if not p.exists():
            errors.append(f"{p}: missing")
            continue
        errs = validate(p)
        errors += errs
        print(f"{'OK ' if not errs else 'BAD'} {p} ({p.stat().st_size:,} bytes)")
        if p == NEURON_TRACE_INDEX:
            bins = sorted(p.parent.glob("*.bin"))
            print(f"    + {len(bins)} binaries ({sum(b.stat().st_size for b in bins):,} bytes)")
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
