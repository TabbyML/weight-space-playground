"""Turn a llama-server /completion response into a token-level recording (pure functions)."""
from __future__ import annotations

from . import answers

THINK_CLOSE = "</think>"
TOKEN_TYPE_CONTROL = 3       # llama.cpp detokenize renders control tokens as '' (special=false)
TOKEN_TYPE_USER_DEFINED = 4  # rendered literally (e.g. "<think>", "</think>")


def _gpt2_unicode_to_byte() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_U2B = _gpt2_unicode_to_byte()


def vocab_piece_bytes(token: str, token_type: int) -> bytes:
    """Raw bytes llama.cpp emits for one vocabulary entry of a byte-level BPE (gpt2-style) vocab."""
    if token_type == TOKEN_TYPE_CONTROL:
        return b""
    if token_type == TOKEN_TYPE_USER_DEFINED:
        return token.encode("utf-8")
    return bytes(_U2B[ch] for ch in token)


def align_logprobs(ids: list[int], probs: list[dict]) -> list[float | None]:
    """Per-token sampled logprob, aligned to the full id sequence.

    llama-server merges tokens that only form a complete UTF-8 character together into a
    single ``completion_probabilities`` entry (carrying the last token's id), so that list
    can be shorter than ``tokens``. Walk both: an entry matches the next id with that id;
    tokens without their own entry get None.
    """
    out: list[float | None] = []
    j = 0
    for tid in ids:
        if j < len(probs) and probs[j]["id"] == tid:
            out.append(probs[j]["logprob"])
            j += 1
        else:
            out.append(None)
    if j != len(probs):
        raise ValueError(f"could not align {len(probs)} probability entries to {len(ids)} tokens")
    return out


def pieces_from_bytes(token_bytes: list[bytes]) -> list[str]:
    """One display string per token whose concatenation is exactly the decoded text.

    Byte-level BPE tokens can split a multi-byte UTF-8 character. Incomplete trailing
    bytes are carried into the next token, so such a token's piece is '' and the
    character appears on the token that completes it. Undecodable bytes left at the
    end are decoded with U+FFFD replacement.
    """
    out: list[str] = []
    carry = b""
    for b in token_bytes:
        buf = carry + b
        # find the longest decodable prefix (an incomplete char is at most 3 bytes)
        for cut in range(len(buf), max(len(buf) - 4, -1), -1):
            try:
                s = buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            out.append(s)
            carry = buf[cut:]
            break
        else:
            out.append(buf.decode("utf-8", errors="replace"))
            carry = b""
    if carry and out:
        out[-1] += carry.decode("utf-8", errors="replace")
    return out


def finish_reason(stop_type: str) -> str:
    """llama-server stop_type -> OpenAI finish_reason as shown on the page."""
    return "length" if stop_type == "limit" else "stop"


def think_end(pieces: list[str]) -> int | None:
    """Index of the ``</think>`` token (tokens before it are thinking), or None if never closed."""
    for i, p in enumerate(pieces):
        if p == THINK_CLOSE:
            return i
    return None


def summarize(pieces: list[str], stop_type: str, reference: str) -> dict:
    te = think_end(pieces)
    answer_text = "".join(pieces[te + 1 :]) if te is not None else ""
    pred = answers.last_boxed(answer_text)
    return {
        "n_tokens": len(pieces),
        "finish": finish_reason(stop_type),
        "thinkEnd": te,
        "answer": pred,
        "correct": answers.is_correct(pred, reference),
    }
