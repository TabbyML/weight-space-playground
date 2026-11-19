from tools.precompute.tokens import finish_reason, pieces_from_bytes, summarize, think_end


def test_pieces_carry_split_utf8():
    text = "a→b"  # → is 3 bytes
    raw = text.encode()
    toks = [raw[:2], raw[2:3], raw[3:]]  # 'a'+first byte, 2nd byte, 3rd byte + 'b'
    pieces = pieces_from_bytes(toks)
    assert len(pieces) == 3
    assert "".join(pieces) == text
    assert pieces[0] == "a" and pieces[1] == ""


def test_pieces_plain_ascii():
    assert pieces_from_bytes([b"Hi", b" there"]) == ["Hi", " there"]


def test_think_end_and_summary():
    pieces = ["Let", " me", " think", "</think>", "\n\n", "\\boxed{", "1048", "}"]
    assert think_end(pieces) == 3
    s = summarize(pieces, "eos", "1048")
    assert s == {"n_tokens": 8, "finish": "stop", "thinkEnd": 3, "answer": "1048", "correct": True}


def test_summary_unclosed_thinking_counts_as_no_answer():
    s = summarize(["\\boxed{1048}"] * 3, "limit", "1048")
    assert s["finish"] == "length" and s["thinkEnd"] is None and s["answer"] is None and not s["correct"]
    assert finish_reason("word") == "stop"


from tools.precompute.tokens import align_logprobs, vocab_piece_bytes  # noqa: E402


def test_vocab_piece_bytes_gpt2_mapping():
    assert vocab_piece_bytes("Ġthe", 1) == b" the"
    assert vocab_piece_bytes("Ċ", 1) == b"\n"
    assert vocab_piece_bytes("â", 1) == bytes([226])
    assert vocab_piece_bytes("</think>", 4) == b"</think>"
    assert vocab_piece_bytes("<|im_end|>", 3) == b""


def test_align_logprobs_merged_utf8_entry():
    ids = [85, 7, 158, 248, 254, 44646]
    probs = [{"id": 85, "logprob": -1.0}, {"id": 7, "logprob": -2.0},
             {"id": 254, "logprob": -3.0}, {"id": 44646, "logprob": -4.0}]
    assert align_logprobs(ids, probs) == [-1.0, -2.0, None, None, -3.0, -4.0]
