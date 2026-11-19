"""Step 1: download every pinned input into the cache (idempotent).

    python -m tools.precompute.fetch [--only bf16|gguf|math]
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import config


def fetch_bf16(which: str) -> Path:
    spec = config.BF16[which]
    return Path(
        hf_hub_download(
            spec["repo"], config.SAFETENSORS_FILE, revision=spec["revision"], cache_dir=config.HF_CACHE
        )
    )


def fetch_gguf(which: str) -> Path:
    dst = config.gguf_path(which)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    print(f"downloading {config.GGUF[which]['url']}")
    with urllib.request.urlopen(config.GGUF[which]["url"]) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dst)
    return dst


def fetch_math500() -> Path:
    spec = config.MATH500
    return Path(
        hf_hub_download(
            spec["repo"], spec["file"], revision=spec["revision"], repo_type="dataset", cache_dir=config.HF_CACHE
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bf16", "gguf", "math"])
    a = ap.parse_args()
    if a.only in (None, "math"):
        print(fetch_math500())
    if a.only in (None, "gguf"):
        for w in config.MODEL_KEYS:
            print(fetch_gguf(w))
    if a.only in (None, "bf16"):
        for w in config.MODEL_KEYS:
            print(fetch_bf16(w))


if __name__ == "__main__":
    main()
