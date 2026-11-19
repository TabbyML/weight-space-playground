"""Step 4+5: summarise the recorded batch, pick demo cases, write replay.json.

    python -m tools.precompute.build_replay [--replay-setting cap16k] [--seed 0] [--max-loop-cases 4]

Inputs: raw recordings in RAW_DIR (from ``record.py``), for every setting in
``config.SETTINGS`` that has recordings. Outputs:
- tools/precompute/out/batch-summary.json  (every recorded candidate x setting, both models; small)
- playground/public/data/replay.json       (chosen cases of ONE setting with full token streams)

The replay setting defaults to ``config.REPLAY_SETTING`` (cap16k, operator decision
2026-11-19). If that setting has no control case, the control is taken from the first of
``CONTROL_FALLBACK`` that has one and is labelled with its own setting/sampling.

Selection rule (deterministic):
- ``loop`` cases: base finish == length AND edited finish == stop AND edited correct.
  The default prompt is taken first if it qualifies; then cases whose base run contains
  a verbatim loop segment (so the UI has a loop moment to highlight); within each group
  by edited token count ascending (shorter edited answers make a faster replay), then id.
- ``control`` case: both models stop and are correct; fewest combined tokens.
"""
from __future__ import annotations

import argparse
import json
import subprocess

from . import config, loops, tokens
from .record import raw_path

LOGPROB_DECIMALS = 2
CONTROL_FALLBACK = ["up16k_pp", "page4096"]
SETTING_NOTE = (
    "Replay uses cap16k (the page's samplers with max_tokens 16384): the strongest, honest contrast. "
    "The upstream-faithful up16k_pp (presence_penalty 1.5) gave weak contrast (cap 90% vs 81%, acc 10% vs 13%) "
    "and reversed the default prompt (base solved it, edited hit the cap); at the page's 4096 cap both models "
    "hit the cap on every prompt. The control case comes from up16k_pp because base never stops in cap16k. "
    "All three batches are in otherSettings."
)
BACKEND_LABEL = {
    "cpu": "CPU (local build of b12211, 1 sequence/server, 4 threads)",
    "vulkan": "Vulkan, RTX 4090 (official b12211 ubuntu-vulkan-x64 release, 4 parallel slots/server)",
}


_VOCAB: dict[str, list[bytes]] = {}


def vocab_bytes(which: str) -> list[bytes]:
    """Per-id raw bytes of the GGUF's vocabulary (cached as JSON in the cache dir)."""
    if which not in _VOCAB:
        cache = config.CACHE / f"vocab-{which}.json"
        if not cache.exists():
            from gguf import GGUFReader

            r = GGUFReader(config.gguf_path(which))
            f, ty = r.fields["tokenizer.ggml.tokens"], r.fields["tokenizer.ggml.token_type"]
            toks = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            types = [int(ty.parts[i][0]) for i in ty.data]
            cache.write_text(json.dumps({"tokens": toks, "types": types}))
        v = json.loads(cache.read_text())
        _VOCAB[which] = [tokens.vocab_piece_bytes(t, ty) for t, ty in zip(v["tokens"], v["types"])]
    return _VOCAB[which]


def load_run(setting: str, which: str, case_id: str, seed: int) -> dict | None:
    p = raw_path(setting, which, case_id, seed)
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    res = rec["response"]
    ids = res["tokens"]
    if len(ids) != res["tokens_predicted"]:
        raise ValueError(f"{p}: {len(ids)} token ids vs tokens_predicted {res['tokens_predicted']}")
    vb = vocab_bytes(which)
    pieces = tokens.pieces_from_bytes([vb[i] for i in ids])
    if "".join(pieces) != res["content"]:
        raise ValueError(f"token pieces do not reproduce content for {p}")
    logprobs = tokens.align_logprobs(ids, res["completion_probabilities"])
    summ = tokens.summarize(pieces, res["stop_type"], rec["case"]["reference"])
    return {
        "case": rec["case"],
        "setting": setting,
        "pieces": pieces,
        "logprobs": [None if lp is None else round(lp, LOGPROB_DECIMALS) for lp in logprobs],
        "loops": loops.find_loops(pieces),
        "wall_s": rec["wall_s"],
        "backend": rec.get("backend", "cpu"),
        "predicted_per_second": res["timings"]["predicted_per_second"],
        **summ,
    }


def batch_stats(rows: list[dict]) -> dict:
    n = len(rows)

    def rate(which: str, key: str, val=True) -> float:
        return round(sum(r[which][key] == val for r in rows) / n, 4) if n else 0.0

    return {
        "n": n,
        "base_cap_rate": rate("base", "finish", "length"),
        "edited_cap_rate": rate("edited", "finish", "length"),
        "base_acc": rate("base", "correct"),
        "edited_acc": rate("edited", "correct"),
    }


def pick_cases(rows: list[dict], max_loop: int) -> tuple[list[str], str | None]:
    loop_ok = [r for r in rows if r["base"]["finish"] == "length" and r["edited"]["finish"] == "stop"
               and r["edited"]["correct"]]
    loop_ok.sort(key=lambda r: (r["id"] != "default", r["base"].get("loopTokens", 0) == 0,
                                r["edited"]["n_tokens"], r["id"]))
    control = [r for r in rows if all(r[w]["finish"] == "stop" and r[w]["correct"] for w in config.MODEL_KEYS)]
    control.sort(key=lambda r: (r["base"]["n_tokens"] + r["edited"]["n_tokens"], r["id"]))
    return [r["id"] for r in loop_ok[:max_loop]], (control[0]["id"] if control else None)


def llama_cpp_version() -> dict:
    commit = subprocess.run(["git", "-C", str(config.LLAMA_CPP), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out = subprocess.run([str(config.LLAMA_SERVER), "--version"], capture_output=True, text=True)
    version = next((l for l in (out.stdout + out.stderr).splitlines() if l.startswith("version")), "")
    return {"commit": commit, "version": version.strip()}


def cpu_model() -> str:
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def run_view(run: dict) -> dict:
    return {
        "tokens": run["pieces"],
        "logprobs": run["logprobs"],
        "thinkEnd": run["thinkEnd"],
        "finish": run["finish"],
        "nTokens": run["n_tokens"],
        "answer": run["answer"],
        "correct": run["correct"],
        "loops": run["loops"],
    }


def summarize_setting(setting: str, cases: list[dict], seed: int, max_loop: int) -> tuple[dict, dict]:
    rows, runs = [], {}
    for c in cases:
        r = {w: load_run(setting, w, c["id"], seed) for w in config.MODEL_KEYS}
        if not all(r.values()):
            continue
        runs[c["id"]] = r
        rows.append({
            "id": c["id"],
            "level": c["level"],
            "reference": c["reference"],
            **{w: {k: r[w][k] for k in ("n_tokens", "finish", "thinkEnd", "answer", "correct")}
               | {"loopTokens": sum(s["end"] - s["start"] for s in r[w]["loops"])}
               for w in config.MODEL_KEYS},
        })
    loop_ids, control_id = pick_cases(rows, max_loop)
    backends = sorted({r[w]["backend"] for r in runs.values() for w in config.MODEL_KEYS})
    summary = {
        "backend": BACKEND_LABEL[backends[0]] if len(backends) == 1 else "mixed: " + ", ".join(backends),
        "sampling": config.SETTINGS[setting]["sampling"],
        "nCtx": config.SETTINGS[setting]["n_ctx"],
        "seed": seed,
        "batchStats": batch_stats(rows),
        "selection": {"loop": loop_ids, "control": control_id},
        "rows": rows,
    }
    return summary, runs


def print_table(setting: str, summ: dict) -> None:
    print(f"\n== {setting} {json.dumps(summ['batchStats'])}")
    print(f"{'id':42s} {'lvl':>3s} | {'base tok':>8s} {'fin':6s} ok loop | {'edit tok':>8s} {'fin':6s} ok loop")
    for r in summ["rows"]:
        b, e = r["base"], r["edited"]
        print(f"{r['id']:42s} {str(r['level']):>3s} | {b['n_tokens']:8d} {b['finish']:6s} {int(b['correct'])} "
              f"{b['loopTokens']:5d} | {e['n_tokens']:8d} {e['finish']:6s} {int(e['correct'])} {e['loopTokens']:5d}")
    print(f"selection: {summ['selection']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--max-loop-cases", type=int, default=4)
    ap.add_argument("--replay-setting", default=config.REPLAY_SETTING, choices=list(config.SETTINGS))
    ap.add_argument("--no-logprobs", action="store_true", help="drop per-token logprobs from replay.json")
    a = ap.parse_args()

    from .candidates import candidates
    from .fetch import fetch_math500

    cases = candidates(fetch_math500())
    summaries, runs = {}, {}
    for setting in config.SETTINGS:
        summ, rr = summarize_setting(setting, cases, a.seed, a.max_loop_cases)
        if summ["rows"]:
            summaries[setting], runs[setting] = summ, rr
            print_table(setting, summ)

    chosen = a.replay_setting
    if chosen not in summaries:
        raise SystemExit(f"no recordings for setting {chosen}")
    summ = summaries[chosen]

    def sampling_of(setting: str) -> dict:
        return summaries[setting]["sampling"] | {"min_p": 0.05}

    source = {
        "engine": "llama.cpp llama-server (native)",
        "llamaCpp": llama_cpp_version(),
        "backend": summ["backend"],
        "host": f"{cpu_model()}; 4x NVIDIA RTX 4090 for Vulkan runs",
        "gguf": {w: config.GGUF[w]["url"] for w in config.MODEL_KEYS},
        "setting": chosen,
        "sampling": sampling_of(chosen),
        "samplingNote": "samplers not listed are at llama-server defaults (repeat_penalty 1.0, no DRY/XTC)",
        "nCtx": summ["nCtx"],
        "seed": a.seed,
        "chatTemplate": "GGUF chat template via llama-server /apply-template, enable_thinking=true",
        "caveat": ("Native llama.cpp recording: a WebGPU (wllama) run with the same seed will diverge token-for-token. "
                   "Base and edited GGUFs are not a matched quant pair (unsloth imatrix Q4_K_M vs plain Q4_K_M)."),
    }

    # Control case: from the chosen setting if it has one, else from the first fallback setting that does
    # (the case then carries that setting's sampling).
    control_setting = next((s for s in [chosen, *CONTROL_FALLBACK] if s in summaries
                            and summaries[s]["selection"]["control"]), None)
    picks = [(chosen, cid, "loop") for cid in summ["selection"]["loop"]]
    if control_setting:
        picks.append((control_setting, summaries[control_setting]["selection"]["control"], "control"))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch = {
        "source": {k: v for k, v in source.items() if k not in ("setting", "sampling", "nCtx", "backend")},
        "replaySetting": chosen,
        "replayCases": [{"setting": st, "id": cid, "kind": kind} for st, cid, kind in picks],
        "settings": summaries,
    }
    (config.OUT_DIR / "batch-summary.json").write_text(json.dumps(batch, indent=1) + "\n")

    out_cases = []
    for setting, cid, kind in picks:
        r = runs[setting][cid]
        c = r["base"]["case"]
        views = {w: run_view(r[w]) for w in config.MODEL_KEYS}
        if a.no_logprobs:
            for v in views.values():
                v.pop("logprobs")
        out_cases.append({
            "id": cid,
            "title": c["title"],
            "prompt": c["prompt"],
            "reference": c["reference"],
            "source": c["source"],
            "kind": kind,
            "setting": setting,
            "sampling": sampling_of(setting),
            "seed": a.seed,
            "backend": summaries[setting]["backend"],
            **views,
        })

    def default_result(setting: str) -> dict:
        row = next((r for r in summaries[setting]["rows"] if r["id"] == "default"), None)
        if not row:
            return {}
        return {w: {k: row[w][k] for k in ("n_tokens", "finish", "correct")} for w in config.MODEL_KEYS}

    replay = {
        "placeholder": False,
        "recorded": True,
        "source": source,
        "batchStats": summ["batchStats"],
        "defaultPrompt": {"qualifies": "default" in summ["selection"]["loop"], **default_result(chosen)},
        "otherSettings": {
            st: {"sampling": sampling_of(st), "backend": summaries[st]["backend"],
                 "batchStats": summaries[st]["batchStats"], "defaultPrompt": default_result(st)}
            for st in summaries if st != chosen
        },
        "settingNote": SETTING_NOTE,
        "cases": out_cases,
    }
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.DATA_DIR / "replay.json"
    dst.write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"\nreplay: {[(st, cid, kind) for st, cid, kind in picks]}")
    print(f"wrote {dst} ({dst.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
