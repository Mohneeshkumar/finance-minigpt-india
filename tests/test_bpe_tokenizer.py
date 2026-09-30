"""Tests for BPETokenizer (Step 9)."""

from src.tokenizer.bpe_tokenizer import BPETokenizer


CORPUS = [
    "the quick brown fox",
    "the quick brown dog",
    "a quick fox jumps",
]


def test_special_tokens_get_reserved_ids():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=60)
    for i, special in enumerate(BPETokenizer.SPECIAL_TOKENS):
        assert tok.token_to_id[special] == i


def test_round_trip_encode_decode():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=60)
    text = "the quick brown fox"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_unseen_word_falls_back_to_unk_without_crashing():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=60)
    ids = tok.encode("zzxxqqyyzz")
    assert isinstance(ids, list) and len(ids) > 0


def test_determinism():
    tok_a = BPETokenizer()
    tok_a.train(CORPUS, vocab_size=60)
    tok_b = BPETokenizer()
    tok_b.train(CORPUS, vocab_size=60)
    assert tok_a.merges == tok_b.merges
    assert tok_a.token_to_id == tok_b.token_to_id


def test_merges_reduce_token_count_vs_raw_characters():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=60)
    text = "the quick brown fox"
    num_chars = sum(len(w) for w in text.split())
    num_tokens = len(tok.encode(text))
    assert num_tokens <= num_chars
