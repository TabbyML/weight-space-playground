from tools.precompute.build_replay import batch_stats, pick_cases
from tools.precompute.validate import Checker, check_run


def row(i, b, e):
    return {"id": i, "base": {"finish": b[0], "correct": b[1], "n_tokens": b[2]},
            "edited": {"finish": e[0], "correct": e[1], "n_tokens": e[2]}}


ROWS = [
    row("default", ("length", False, 100), ("length", False, 100)),
    row("a", ("length", False, 100), ("stop", True, 50)),
    row("b", ("length", False, 100), ("stop", True, 20)),
    row("c", ("length", False, 100), ("stop", False, 10)),   # edited wrong: not a loop case
    row("d", ("stop", True, 30), ("stop", True, 30)),
    row("e", ("stop", True, 10), ("stop", True, 15)),
]


def test_pick_cases_rule():
    loop, control = pick_cases(ROWS, max_loop=4)
    assert loop == ["b", "a"]          # edited length ascending
    assert control == "e"              # fewest combined tokens


def test_pick_cases_default_first_when_it_qualifies():
    rows = ROWS + [row("default", ("length", False, 100), ("stop", True, 99))]
    rows = [r for r in rows[1:]]
    assert pick_cases(rows, 1)[0] == ["default"]


def test_batch_stats():
    s = batch_stats(ROWS)
    assert s["n"] == 6
    assert s["base_cap_rate"] == round(4 / 6, 4)
    assert s["edited_acc"] == round(4 / 6, 4)


def test_validate_check_run():
    good = {"tokens": ["a", "</think>", "b"], "logprobs": [0, 0, 0], "nTokens": 3, "finish": "stop",
            "thinkEnd": 1, "correct": True, "answer": "b", "loops": []}
    c = Checker("x")
    check_run(good, c, "r")
    assert c.errors == []
    bad = dict(good, thinkEnd=0, loops=[{"start": 2, "end": 1, "period": 1, "repeats": 1}])
    c = Checker("x")
    check_run(bad, c, "r")
    assert len(c.errors) == 2


def test_pick_cases_prefers_visible_base_loop():
    rows = [row("x", ("length", False, 100), ("stop", True, 10)),
            row("y", ("length", False, 100), ("stop", True, 50))]
    rows[1]["base"]["loopTokens"] = 40
    rows[0]["base"]["loopTokens"] = 0
    assert pick_cases(rows, 2)[0] == ["y", "x"]
