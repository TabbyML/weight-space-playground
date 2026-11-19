"""Loop (repeated-span) detection over a token sequence (pure functions).

Algorithm (kept deliberately simple, O(n * max_period)):

A *loop segment* is a maximal run ``tokens[start:end]`` in which every token equals the
token ``period`` positions earlier, i.e. the sequence is periodic with that period over
the run. We scan periods ``p = min_period .. max_period``; for each p we compute runs of
``tokens[i] == tokens[i - p]`` (the run plus its first period is the periodic span).
A span qualifies if it covers at least ``min_repeats`` full periods (span >= repeats * p)
and at least ``min_len`` tokens. Spans for all periods are then greedily selected by
length (longest first), dropping spans that overlap an already selected one; for equal
length the smaller period wins (the fundamental period, not a multiple of it).

Each segment is reported as ``{start, end, period, repeats}`` with ``end`` exclusive and
``repeats = (end - start) / period`` (float, 2 decimals), so the UI can highlight
``[start, end)`` and show a "xN" counter.
"""
from __future__ import annotations

from collections.abc import Sequence


def periodic_spans(tokens: Sequence, period: int) -> list[tuple[int, int]]:
    """Maximal spans [start, end) where tokens[i] == tokens[i - period] for start+period <= i < end."""
    spans: list[tuple[int, int]] = []
    n = len(tokens)
    i = period
    while i < n:
        if tokens[i] == tokens[i - period]:
            j = i
            while j < n and tokens[j] == tokens[j - period]:
                j += 1
            spans.append((i - period, j))
            i = j
        else:
            i += 1
    return spans


def find_loops(
    tokens: Sequence,
    min_period: int = 4,
    max_period: int = 400,
    min_repeats: float = 3.0,
    min_len: int = 48,
) -> list[dict]:
    """Non-overlapping periodic segments, sorted by start. See module docstring."""
    cands: list[tuple[int, int, int]] = []  # (start, end, period)
    for p in range(min_period, min(max_period, len(tokens) // 2) + 1):
        for s, e in periodic_spans(tokens, p):
            if e - s >= max(min_len, min_repeats * p):
                cands.append((s, e, p))
    cands.sort(key=lambda c: (-(c[1] - c[0]), c[2], c[0]))
    chosen: list[tuple[int, int, int]] = []
    for s, e, p in cands:
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, p))
    chosen.sort()
    return [
        {"start": s, "end": e, "period": p, "repeats": round((e - s) / p, 2)} for s, e, p in chosen
    ]


def loop_coverage(tokens: Sequence, segments: list[dict]) -> float:
    """Fraction of tokens covered by loop segments."""
    if not tokens:
        return 0.0
    return sum(seg["end"] - seg["start"] for seg in segments) / len(tokens)
