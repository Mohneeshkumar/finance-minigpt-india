"""
Byte-Pair Encoding (BPE) Tokenizer, trained from scratch on a custom corpus.
Reference: Sennrich et al. (2015).
"""

import re
from collections import Counter


class BPETokenizer:
    END_OF_WORD = "</w>"
    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def _word_to_symbols(self, word: str) -> list[str]:
        return list(word) + [self.END_OF_WORD]

    def _get_pair_counts(self, corpus_symbols: list[list[str]]) -> Counter:
        pairs = Counter()
        for symbols in corpus_symbols:
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_pair(self, symbols: list[str], pair: tuple[str, str]) -> list[str]:
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def train(self, corpus: list[str], vocab_size: int) -> None:
        words = re.findall(r"\S+", " ".join(corpus))
        corpus_symbols = [self._word_to_symbols(w) for w in words]

        vocab = list(self.SPECIAL_TOKENS)
        seen_chars = sorted({sym for symbols in corpus_symbols for sym in symbols})
        vocab.extend(seen_chars)

        while len(vocab) < vocab_size:
            pair_counts = self._get_pair_counts(corpus_symbols)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)

            corpus_symbols = [self._merge_pair(s, best_pair) for s in corpus_symbols]
            vocab.append(best_pair[0] + best_pair[1])

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def _tokenize_word(self, word: str) -> list[str]:
        symbols = self._word_to_symbols(word)
        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)
        return symbols

    def encode(self, text: str) -> list[int]:
        unk_id = self.token_to_id["<UNK>"]
        ids = []
        for word in re.findall(r"\S+", text):
            for symbol in self._tokenize_word(word):
                ids.append(self.token_to_id.get(symbol, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        text = "".join(tokens)
        text = text.replace(self.END_OF_WORD, " ")
        return text.strip()
