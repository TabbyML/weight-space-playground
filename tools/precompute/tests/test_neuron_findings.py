import numpy as np

from tools.precompute.neuron_findings import (
    channel_shares,
    effect_size,
    loop_mask,
    n_for_share,
    union_loop_mask,
    window_mean,
)


def test_loop_mask():
    m = loop_mask(10, [{"start": 2, "end": 4}, {"start": 7, "end": 10}])
    assert m.tolist() == [False, False, True, True, False, False, False, True, True, True]


def test_window_mean_clips_and_empty():
    x = np.arange(10.0)
    assert window_mean(x, 5, -2, 0) == 3.5
    assert window_mean(x, 1, -5, 1) == 0.5  # clipped to [0, 2)
    assert window_mean(x, 9, 5, 10) is None


def test_effect_size_sign():
    assert effect_size(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0])) > 0
    assert effect_size(np.ones(3), np.ones(3)) == 0.0


def test_channel_shares_and_n_for_share():
    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    sh = channel_shares(a, np.array([6.0, -3.0, 1.0]))
    assert np.allclose(sh, [0.6, 0.3, 0.1])
    assert n_for_share(sh, 0.5) == 1
    assert n_for_share(sh, 0.8) == 2
    assert n_for_share(sh, 1.0) == 3


def test_union_loop_mask_finds_overlapping_spans():
    # period-4 repetition long enough to qualify (>= 48 tokens and >= 3 periods)
    toks = list("xyz") + list("abcd") * 20 + list("qrs")
    m = union_loop_mask(toks)
    assert m[3:83].all() and not m[:3].any() and not m[83:].any()
