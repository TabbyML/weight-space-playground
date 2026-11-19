"""Pinned inputs and paths for the offline precompute pipeline.

Every remote input is pinned to an immutable revision so reruns are reproducible.
Large artifacts live under CACHE (never in the repo); small outputs are written
into the repo (OUT_DIR, DATA_DIR).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("WE_CACHE", Path.home() / ".cache" / "weight-editing"))
HF_CACHE = CACHE / "hf"
GGUF_DIR = CACHE / "gguf"
RAW_DIR = CACHE / "recordings"
LLAMA_CPP = CACHE / "llama.cpp"
LLAMA_SERVER = LLAMA_CPP / "build" / "bin" / "llama-server"  # CPU build of the commit below
# Official prebuilt Vulkan release of the same commit (b12211 = 7d7f02875), for the GPU batches.
LLAMA_VULKAN_RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/b12211/llama-b12211-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_SERVER_VULKAN = CACHE / "llama-vulkan" / "llama-b12211" / "llama-server"

OUT_DIR = REPO / "tools" / "precompute" / "out"
DATA_DIR = REPO / "playground" / "public" / "data"

# bf16 safetensors (the source of truth for the weight diff).
BF16 = {
    "base": {"repo": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
    "edited": {
        "repo": "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        "revision": "d1671a178705bcab7644806d23c535517f35723b",
    },
}
SAFETENSORS_FILE = "model.safetensors-00001-of-00001.safetensors"

# Must match MODELS in playground/src/main.js exactly (same URLs the page loads).
GGUF = {
    "base": {
        "repo": "unsloth/Qwen3.5-0.8B-GGUF",
        "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/"
        "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf",
    },
    "edited": {
        "repo": "NotCherry/Qwen3.5-0.8B-weight-space-programming-GGUF",
        "url": "https://huggingface.co/NotCherry/Qwen3.5-0.8B-weight-space-programming-GGUF/resolve/"
        "aded4c8fa22992e397cdad51a42299abf463ba52/Qwen3.5-0.8B-Weight-Space-Programming-Q4_K_M.gguf",
    },
}

MATH500 = {"repo": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "file": "test.jsonl"}

DEFAULT_PROMPT = (
    "What is $2^{10} + 24$? Please reason step by step, and put your final answer within \\boxed{}."
)
DEFAULT_REFERENCE = "1048"

# Recording settings. Samplers not listed (min_p 0.05, repeat penalty 1.0, ...) stay at the
# llama.cpp server defaults, as on the page.
#   page4096  - the page's current live defaults (playground/src/main.js / index.html)
#   up16k_pp  - upstream-faithful (exp8 eval used presence_penalty 1.5), cap raised to 16384
#   cap16k    - cap 16384 without presence penalty (separates the cap effect from the penalty)
_BASE_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
SETTINGS = {
    "page4096": {"sampling": _BASE_SAMPLING | {"presence_penalty": 0.0, "max_tokens": 4096}, "n_ctx": 8192},
    "up16k_pp": {"sampling": _BASE_SAMPLING | {"presence_penalty": 1.5, "max_tokens": 16384}, "n_ctx": 20480},
    "cap16k": {"sampling": _BASE_SAMPLING | {"presence_penalty": 0.0, "max_tokens": 16384}, "n_ctx": 20480},
}
DEFAULT_SETTING = "page4096"
# Setting that feeds replay.json (decision 2026-11-19: strongest honest contrast; see README).
REPLAY_SETTING = "cap16k"
SEED = 0

MODEL_KEYS = ("base", "edited")


def gguf_path(which: str) -> Path:
    return GGUF_DIR / GGUF[which]["url"].split("/")[-1]
