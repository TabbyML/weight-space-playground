import json

import numpy as np
import pytest

from tools.precompute import validate
from tools.precompute.neuron_trace import (
    Packer,
    bin_edges,
    binned_mean_abs,
    channel_summary,
    forward_ids,
    parse_llama_tokenize_ids,
    push_weights,
    quantize_int8,
    quantize_uint8,
    run_record,
    sig,
)


def test_bin_edges_cover_all_tokens():
    for n in (1, 5, 255, 256, 257, 4584, 16384):
        e = bin_edges(n)
        assert e[0] == 0 and e[-1] == n
        assert len(e) - 1 == min(n, 256)
        assert all(a < b for a, b in zip(e, e[1:]))  # every bin non-empty
    with pytest.raises(ValueError):
        bin_edges(0)


def test_binned_mean_abs():
    a = np.array([[1.0, -2.0], [-3.0, 4.0], [5.0, 0.0]])
    h = binned_mean_abs(a, [0, 2, 3])
    assert h.shape == (2, 2)
    assert np.allclose(h, [[2.0, 5.0], [3.0, 0.0]])


def test_quantize_int8_roundtrip_bound():
    x = np.random.default_rng(0).normal(size=1000) * 3
    q, s = quantize_int8(x)
    assert q.dtype == np.int8 and np.abs(q).max() == 127
    assert np.isclose(s * 127, np.abs(x).max())
    assert np.abs(q * s - x).max() <= s / 2 + 1e-12
    assert quantize_int8(np.zeros(4))[1] == 0.0


def test_quantize_uint8():
    x = np.array([0.0, 0.5, 2.0])
    q, s = quantize_uint8(x)
    assert q.dtype == np.uint8 and q.tolist() == [0, 64, 255]
    with pytest.raises(ValueError):
        quantize_uint8(np.array([-1.0]))


def test_push_weights_match_rank1_delta():
    rng = np.random.default_rng(1)
    u = rng.normal(size=16)
    u /= np.linalg.norm(u)
    coeff = rng.normal(size=4)
    delta = np.outer(u, coeff)
    w = push_weights(u, delta)
    assert np.allclose(w, coeff)
    a = rng.normal(size=(7, 4))
    assert np.allclose(a @ w, (delta @ a.T).T @ u)  # push = u . (Delta a)


def test_forward_ids_alignment():
    ids, start = forward_ids([10, 11, 12], [20, 21, 22])
    assert ids == [10, 11, 12, 20, 21]
    assert start == 2  # logits at position 2 (last prompt token) sampled gen[0]
    assert len(ids) - start == 3  # one position per generated token
    with pytest.raises(ValueError):
        forward_ids([], [1])


def test_parse_llama_tokenize_ids():
    out = "some log line\n[248045, 846, 198]\n"
    assert parse_llama_tokenize_ids(out) == [248045, 846, 198]
    with pytest.raises(ValueError):
        parse_llama_tokenize_ids("nothing here")


def test_sig():
    assert sig(1234.5) == 1230.0
    assert sig(-0.0012345) == -0.00123
    assert sig(0) == 0.0


def test_run_record_and_channel_summary():
    a = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    w = {15: np.array([0.5, -1.0]), 19: np.array([1.0, 1.0])}
    arrs = {"a15": a, "proj15": np.array([0.1, 0.2]), "a19": a, "proj19": np.array([0.3, 0.4])}
    r = run_record(arrs, w)
    assert np.allclose(r[15]["push"], [-1.5, 5.5])
    assert np.allclose(r[19]["norm"], [np.sqrt(5), 5.0])
    cs = channel_summary(a, w[15])
    assert cs["meanAbs"] == [2.0, 3.0] and cs["meanPush"] == [1.0, 1.0]


def _synthetic_index(tmp_path, corrupt=False):
    """Tiny neuron-trace/index.json + .bin + replay.json, built the same way as build_files."""
    rng = np.random.default_rng(2)
    n = {"base": 300, "edited": 120}
    replay = {"cases": [{"id": "c1", "base": {"nTokens": n["base"]}, "edited": {"nTokens": n["edited"]}}]}
    (tmp_path / "replay.json").write_text(json.dumps(replay))
    d = tmp_path / "neuron-trace"
    d.mkdir()
    pk = Packer()
    runs = {}
    base15 = {"push": rng.normal(size=n["base"]), "proj": rng.normal(size=n["base"])}
    for run, (wts, stream) in {"base": ("base", "base"), "edited": ("edited", "edited"),
                               "counterfactual": ("edited", "base")}.items():
        nt = n[stream]
        edges = bin_edges(nt)
        layers = {}
        for L in (15, 19):
            if run == "counterfactual" and L == 15:
                layers["15"] = {"push": {"sameAs": "base"}, "norm": {"sameAs": "base"},
                                "proj": pk.series(base15["proj"] + base15["push"]),
                                "heat": {"sameAs": "base"}, "channels": {"sameAs": "base"}}
                continue
            a = rng.normal(size=(nt, 64))
            push = base15["push"] if (run, L) == ("base", 15) else rng.normal(size=nt)
            proj = base15["proj"] if (run, L) == ("base", 15) else rng.normal(size=nt)
            layers[str(L)] = {"push": pk.series(push), "proj": pk.series(proj),
                              "norm": pk.series(np.linalg.norm(a, axis=1)),
                              "heat": pk.heat(binned_mean_abs(a, edges)),
                              "channels": channel_summary(a, np.ones(64))}
        runs[run] = {"weights": wts, "tokens": stream, "nTokens": nt, "binEdges": edges, "layers": layers}
    blob = pk.blob()
    if corrupt:
        blob = blob[:-1]
    (d / "c1.bin").write_bytes(blob)
    index = {"placeholder": False, "source": "bf16", "formatVersion": 1, "engine": "x", "alignment": "x",
             "quantized": "x", "layers": [15, 19],
             "edit": {str(L): {"columns": list(range(64)), "pushWeights": [1.0] * 64} for L in (15, 19)},
             "runs": {"base": {"weights": "base", "tokens": "base"},
                      "edited": {"weights": "edited", "tokens": "edited"},
                      "counterfactual": {"weights": "edited", "tokens": "base"}},
             "cases": [{"id": "c1", "file": "c1.bin", "bytes": len(pk.blob()), "runs": runs}]}
    (d / "index.json").write_text(json.dumps(index))
    return d / "index.json"


def test_validate_accepts_packed_index(tmp_path):
    assert validate.validate(_synthetic_index(tmp_path)) == []


def test_validate_rejects_truncated_binary(tmp_path):
    errs = validate.validate(_synthetic_index(tmp_path, corrupt=True))
    assert any("size != bytes" in e for e in errs)
