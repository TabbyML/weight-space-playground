from tools.precompute.loops import find_loops, loop_coverage, periodic_spans


def test_periodic_spans_basic():
    seq = list("xyabcabcabcz")
    # period 3 run: 'abcabcabc' starts at 2, ends at 11
    assert (2, 11) in periodic_spans(seq, 3)


def test_find_loops_reports_fundamental_period():
    prefix = [f"p{i}" for i in range(50)]
    unit = ["Wait", ",", " is", " it", " right", "?", "\n"]
    seq = prefix + unit * 20 + ["end"]
    segs = find_loops(seq, min_period=2, min_len=10)
    assert len(segs) == 1
    s = segs[0]
    assert s["period"] == 7
    assert s["start"] == 50
    assert s["end"] == 50 + 7 * 20
    assert s["repeats"] == 20.0


def test_find_loops_ignores_short_or_nonrepeating():
    assert find_loops([str(i) for i in range(500)]) == []
    # only two repeats of a long unit: below min_repeats
    unit = [f"u{i}" for i in range(30)]
    assert find_loops(unit * 2, min_len=10) == []


def test_find_loops_two_separate_segments_and_coverage():
    a = ["a1", "a2", "a3", "a4"] * 15
    b = ["b1", "b2", "b3", "b4", "b5"] * 12
    seq = ["s"] + a + [f"m{i}" for i in range(20)] + b
    segs = find_loops(seq, min_len=40)
    assert [(s["period"], s["start"]) for s in segs] == [(4, 1), (5, 81)]
    assert 0 < loop_coverage(seq, segs) <= 1
