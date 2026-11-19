"""Candidate prompts for the recording batch (deterministic).

The page's default prompt plus a fixed, difficulty-stratified sample of MATH-500
problems with plain-integer reference answers (so grading is unambiguous). The
selection is a pure function of the pinned dataset: problems are ordered by
sha1(unique_id) and the first N per level are taken.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import config

SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
PER_LEVEL = {1: 8, 2: 8, 3: 7, 4: 4, 5: 3}
MAX_PROBLEM_CHARS = 400


def case_id_for(unique_id: str) -> str:
    """'test/algebra/2036.json' -> 'math500-algebra-2036'."""
    parts = unique_id.removesuffix(".json").split("/")
    return "math500-" + "-".join(parts[1:])


def select(rows: list[dict], per_level: dict[int, int] = PER_LEVEL) -> list[dict]:
    eligible = [
        r for r in rows
        if re.fullmatch(r"-?\d+", r["answer"].strip())
        and len(r["problem"]) <= MAX_PROBLEM_CHARS
        and "[asy]" not in r["problem"]
    ]
    eligible.sort(key=lambda r: hashlib.sha1(r["unique_id"].encode()).hexdigest())
    out: list[dict] = []
    for level, n in sorted(per_level.items()):
        out += [r for r in eligible if r["level"] == level][:n]
    return out


def candidates(math_path) -> list[dict]:
    rows = [json.loads(line) for line in open(math_path)]
    cases = [{
        "id": "default",
        "title": "Default prompt (2^10 + 24)",
        "prompt": config.DEFAULT_PROMPT,
        "reference": config.DEFAULT_REFERENCE,
        "source": "page default",
        "level": None,
    }]
    for r in select(rows):
        cases.append({
            "id": case_id_for(r["unique_id"]),
            "title": f"MATH-500 {r['subject']} L{r['level']}",
            "prompt": r["problem"].strip() + SUFFIX,
            "reference": r["answer"].strip(),
            "source": f"HuggingFaceH4/MATH-500 {r['unique_id']}",
            "level": r["level"],
        })
    return cases
