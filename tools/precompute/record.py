"""Step 3: record base and edited runs on the pinned GGUFs with native llama.cpp (CPU).

    python -m tools.precompute.record [--settings page4096 ...] [--servers-base 8]
        [--servers-edited 8] [--threads 4] [--seeds 0] [--only CASE_ID ...]
        [--backend cpu|vulkan] [--gpus 0 1] [--slots 4]

Backend ``cpu`` (default) starts single-slot llama-server processes per model (every run
is a single sequence with a fixed thread count, i.e. reproducible). Backend ``vulkan``
starts ``--servers-base``/``--servers-edited`` llama-server processes (all layers
offloaded, ``--slots`` parallel sequences each), spread round-robin over ``--gpus``.
Several small servers per GPU beat one big one: a single server process tops out at
~200 tok/s aggregate (host-side bound) while one 4090 handles several. Continuous
batching makes GPU runs not bit-reproducible across reruns. Each request renders each prompt with the model's own
chat template (``enable_thinking: true``, as the page does), and calls ``/completion``
with the sampling of each requested setting (``config.SETTINGS``). Each run is cached as
``RAW_DIR/<setting>/<model>/<case>__s<seed>.json``; existing files are skipped (idempotent).
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import candidates, config
from .fetch import fetch_math500

N_PROBS = 5
BACKEND = ["cpu"]  # set from --backend in main(); recorded into every raw file


def _post(port: int, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_ready(port: int, proc: subprocess.Popen, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server on {port} exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(port)


def raw_path(setting: str, which: str, case_id: str, seed: int) -> Path:
    return config.RAW_DIR / setting / which / f"{case_id}__s{seed}.json"


def record_one(port: int, which: str, case: dict, setting: str, seed: int) -> dict:
    prompt = _post(port, "/apply-template", {
        "messages": [{"role": "user", "content": case["prompt"]}],
        "chat_template_kwargs": {"enable_thinking": True},
    })["prompt"]
    s = config.SETTINGS[setting]["sampling"]
    t0 = time.time()
    res = _post(port, "/completion", {
        "prompt": prompt,
        "n_predict": s["max_tokens"],
        "temperature": s["temperature"],
        "top_p": s["top_p"],
        "top_k": s["top_k"],
        "presence_penalty": s["presence_penalty"],
        "seed": seed,
        "n_probs": N_PROBS,
        "return_tokens": True,
        "cache_prompt": False,
    })
    return {
        "case": case,
        "model": which,
        "setting": setting,
        "backend": BACKEND[0],
        "sampling": s,
        "gguf_url": config.GGUF[which]["url"],
        "seed": seed,
        "rendered_prompt": prompt,
        "wall_s": round(time.time() - t0, 2),
        "response": res,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=[config.DEFAULT_SETTING], choices=list(config.SETTINGS))
    ap.add_argument("--servers-base", type=int, default=8)
    ap.add_argument("--servers-edited", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--port-base", type=int, default=18100)
    ap.add_argument("--slots", type=int, default=32, help="parallel slots per server (vulkan)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[config.SEED])
    ap.add_argument("--only", nargs="*", help="case ids to (re)record")
    a = ap.parse_args()
    BACKEND[0] = a.backend

    cases = candidates.candidates(fetch_math500())
    if a.only:
        cases = [c for c in cases if c["id"] in a.only]
    (config.OUT_DIR).mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    jobs: dict[str, queue.Queue] = {w: queue.Queue() for w in config.MODEL_KEYS}
    for w in config.MODEL_KEYS:
        for setting in a.settings:
            for seed in a.seeds:
                for c in cases:
                    if not raw_path(setting, w, c["id"], seed).exists():
                        jobs[w].put((c, setting, seed))
    total = sum(q.qsize() for q in jobs.values())
    n_ctx = max(config.SETTINGS[s]["n_ctx"] for s in a.settings)
    print(f"{len(cases)} cases x {len(a.settings)} settings x {len(a.seeds)} seeds; {total} runs to record",
          flush=True)
    if not total:
        return

    lock = threading.Lock()
    done = [0]

    def worker(port: int, which: str) -> None:
        q = jobs[which]
        while True:
            try:
                case, setting, seed = q.get_nowait()
            except queue.Empty:
                return
            rec = record_one(port, which, case, setting, seed)
            dst = raw_path(setting, which, case["id"], seed)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.rename(dst)
            with lock:
                done[0] += 1
                r = rec["response"]
                print(f"[{done[0]}/{total}] {which:6s} {setting:8s} {case['id']:40s} s{seed} "
                      f"{r['tokens_predicted']:5d} tok {r['stop_type']:5s} {rec['wall_s']}s", flush=True)

    port = a.port_base
    gpu_rr = [0]
    threads = []
    try:
        for w in config.MODEL_KEYS:
            if not jobs[w].qsize():
                continue
            n_servers = min(a.servers_base if w == "base" else a.servers_edited, jobs[w].qsize())
            exe = config.LLAMA_SERVER if a.backend == "cpu" else config.LLAMA_SERVER_VULKAN
            for _ in range(n_servers):
                if a.backend == "cpu":
                    n_workers, ctx = 1, n_ctx
                    extra = ["-np", "1", "-t", str(a.threads)]
                else:
                    n_workers, ctx = a.slots, n_ctx * a.slots
                    gpu = a.gpus[gpu_rr[0] % len(a.gpus)]
                    gpu_rr[0] += 1
                    extra = ["-np", str(n_workers), "-ngl", "99", "--device", f"Vulkan{gpu}", "-t", "2"]
                port += 1
                log = open(config.RAW_DIR.parent / f"server-{port}.log", "w")
                p = subprocess.Popen([
                    str(exe), "-m", str(config.gguf_path(w)), "-c", str(ctx), *extra,
                    "--port", str(port), "--host", "127.0.0.1", "--no-webui",
                ], stdout=log, stderr=subprocess.STDOUT)
                procs.append(p)
                _wait_ready(port, p, timeout=300)
                for _ in range(n_workers):
                    t = threading.Thread(target=worker, args=(port, w))
                    t.start()
                    threads.append(t)
        for t in threads:
            t.join()
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
