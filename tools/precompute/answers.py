"""Final-answer extraction and grading (pure functions).

Deliberately simple: take the last ``\\boxed{...}`` (brace-matched) from the
*answer* part of a completion, normalise a few LaTeX spellings, and compare as
strings, falling back to a numeric comparison. Candidates are chosen with simple
reference answers (integers, see ``candidates.py``) so this is sufficient.
"""
from __future__ import annotations

import re
from fractions import Fraction


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` / ``\\fbox{...}`` in ``text`` (brace-matched)."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            best = text[m.end() : i - 1]
    return best


def normalize(ans: str) -> str:
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("$", ""), ("^\\circ", ""),
                 ("^{\\circ}", ""), ("\\%", ""), ("%", "")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # "x = 5" -> "5"
    if re.fullmatch(r"[a-zA-Z]\w*=.+", s):
        s = s.split("=", 1)[1]
    # thousands separators: 1,048 / 1{,}048
    s = s.replace("{,}", ",")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _as_number(s: str) -> Fraction | None:
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s) or re.fullmatch(r"(-?)\\frac(\d)(\d)", s)
    if m:
        v = Fraction(int(m.group(2)), int(m.group(3)))
        return -v if m.group(1) else v
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def is_correct(pred: str | None, reference: str) -> bool:
    if pred is None:
        return False
    p, r = normalize(pred), normalize(reference)
    if p == r:
        return True
    pn, rn = _as_number(p), _as_number(r)
    return pn is not None and rn is not None and pn == rn
