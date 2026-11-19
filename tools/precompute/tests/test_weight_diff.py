import numpy as np

from tools.precompute.weight_diff import bf16_bits_to_f32, changed_axes, f16_round, rank_structure, summarize_tensor


def test_bf16_widening_exact():
    x = np.array([1.0, -2.5, 0.15625], dtype=np.float32)
    bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(bf16_bits_to_f32(bits), x)


def test_changed_axes_columns():
    d = np.zeros((4, 6))
    d[:, [1, 4]] = 1.0
    rows, cols = changed_axes(d)
    assert rows.tolist() == [0, 1, 2, 3]
    assert cols.tolist() == [1, 4]


def test_rank_structure_recovers_rank1_edit():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(32, 20))
    u = rng.normal(size=32)
    u /= np.linalg.norm(u)
    cols = [2, 5, 7, 11]
    alpha = 6.0
    edited = base.copy()
    for j in cols:
        edited[:, j] -= alpha * (u @ base[:, j]) * u
    s = summarize_tensor("t", base, edited)
    assert s["changedColumns"] == cols
    assert s["svd"]["rank1EnergyFraction"] > 0.999999
    assert abs(abs(s["svd"]["u"] @ u) - 1) < 1e-9
    assert abs(s["alphaMean"] - alpha) < 1e-9
    assert np.allclose(s["projAfter"], (1 - alpha) * s["projBefore"])
    # sign convention: majority of coefficients negative
    assert np.sum(s["svd"]["coeff"] < 0) >= np.sum(s["svd"]["coeff"] > 0)


def test_rank_structure_direct():
    d = np.outer([1.0, 0.0], [-2.0, -1.0])
    rs = rank_structure(d)
    assert np.allclose(np.outer(rs["u"], rs["coeff"]), d)


def test_f16_round():
    assert f16_round([0.1]) == [0.099976]
