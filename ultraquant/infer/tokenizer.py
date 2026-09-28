"""The text boundary, read from the same GGUF as the weights.

256,000 spellings are not a vocabulary to reconstruct from a model
name. The file supplies the tokens, their types, the merge order,
and the special-token policy. Only the algorithm lives here: GPT-2's
byte alphabet and Command-R's two pre-tokenisation passes.

**The order is part of the model.** Unicode numbers are isolated
first, one code point at a time. The case-sensitive GPT-2 pattern
then splits each remaining fragment. Combining those passes into
one expression changes where spaces land, and produces plausible
token IDs which are nevertheless the wrong input to the weights.
Unicode categories come from `unicodedata`; no tokenizer package,
downloaded vocabulary, or third-party regular expression is needed.

**Bytes survive the boundary.** BPE works on UTF-8 bytes represented
by reversible Unicode spellings. Decoding joins those bytes before
decoding UTF-8, because a token can end halfway through a character.
Special tokens retain their literal spelling, including an added
BOS; `decode(encode(text, add_special=False))` preserves the text.
Arbitrary IDs that form incomplete UTF-8 use replacement characters.

This is the `gpt2` / `command-r` tokenizer, not a guess at every GGUF
tokenizer family. Other algorithms are refused by name. Merge
results are cached per fragment, with a bound so a long conversation
does not keep every word it has ever seen.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path

from ultraquant.convert import gguf

__all__ = ["Tokenizer"]


def _byte_alphabet() -> tuple[str, ...]:
    """GPT-2's printable bytes, followed by the missing bytes in order."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {byte: chr(byte) for byte in visible}
    for byte in range(256):
        if byte not in alphabet:
            alphabet[byte] = chr(256 + len(alphabet) - len(visible))
    return tuple(alphabet[byte] for byte in range(256))


_BYTE_TO_CHAR = _byte_alphabet()
_CHAR_TO_BYTE = {char: byte for byte, char in enumerate(_BYTE_TO_CHAR)}
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _kind(char: str) -> str:
    category = unicodedata.category(char)[0]
    if category in "LN":
        return category
    # Unicode White_Space, unlike str.isspace(), excludes U+001C..001F.
    if char.isspace() and char not in "\x1c\x1d\x1e\x1f":
        return "S"
    return "P"


def _gpt2_fragments(text: str, kinds: list[str], start: int, end: int):
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S).

    Unmatched whitespace is retained, as in llama.cpp's regex splitter.
    `end` is a first-pass boundary, so lookahead cannot cross a number.
    """
    at = start
    while at < end:
        first = at
        contraction = next((s for s in _CONTRACTIONS
                            if text.startswith(s, at, end)), None) if text[at] == "'" else None
        if contraction is not None:
            at += len(contraction)
        else:
            word = at + (text[at] == " " and at + 1 < end)
            kind = kinds[word]
            if kind in "LNP":
                at = word + 1
                while at < end and kinds[at] == kind:
                    at += 1
            else:
                at += 1
                while at < end and kinds[at] == "S":
                    at += 1
                if at < end and at - first > 1:
                    at -= 1
        yield text[first:at]


def _command_r_fragments(text: str):
    """Apply \\p{N} first, then the GPT-2 expression to each fragment."""
    kinds = [_kind(char) for char in text]
    start = 0
    for at, kind in enumerate(kinds):
        if kind == "N":
            yield from _gpt2_fragments(text, kinds, start, at)
            yield text[at]
            start = at + 1
    yield from _gpt2_fragments(text, kinds, start, len(text))


class Tokenizer:
    """A checkpoint's byte-level BPE vocabulary and special-token policy."""

    @classmethod
    def from_gguf(cls, path: str | Path) -> Tokenizer:
        opened = gguf.read(path, keep=("tokenizer.ggml.tokens",
                                      "tokenizer.ggml.merges"))
        return cls(opened.metadata)

    def __init__(self, metadata: dict):
        prefix = "tokenizer.ggml."
        model, pre = metadata[prefix + "model"], metadata[prefix + "pre"]
        if (model, pre) != ("gpt2", "command-r"):
            raise ValueError(f"unsupported tokenizer {model!r} / {pre!r}; "
                             "expected 'gpt2' / 'command-r'")
        tokens = metadata[prefix + "tokens"]
        merges = metadata[prefix + "merges"]
        types = metadata[prefix + "token_type"]
        if not isinstance(tokens, list) or not isinstance(merges, list):
            raise ValueError("tokenizer tokens and merges must be retained string arrays")
        if len(types) != len(tokens):
            raise ValueError("token types must have one entry per vocabulary token")
        self.tokens = tuple(tokens)
        self.token_types = tuple(types)
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._ranks = {}
        for rank, merge in enumerate(merges):
            pair = tuple(merge.split(" "))
            if len(pair) != 2 or not all(pair):
                raise ValueError(f"invalid BPE merge at rank {rank}: {merge!r}")
            self._ranks.setdefault(pair, rank)
        self.add_bos = bool(metadata[prefix + "add_bos_token"])
        self.add_eos = bool(metadata[prefix + "add_eos_token"])
        self.bos_id = metadata.get(prefix + "bos_token_id")
        self.eos_id = metadata.get(prefix + "eos_token_id")
        for name, enabled, token_id in (("BOS", self.add_bos, self.bos_id),
                                         ("EOS", self.add_eos, self.eos_id)):
            if enabled and (not isinstance(token_id, int)
                            or not 0 <= token_id < len(tokens)):
                raise ValueError(f"missing or invalid {name} token ID")
        self._special = {token: index for index, (token, kind)
                         in enumerate(zip(tokens, types)) if kind in (2, 3, 4) and token}
        self._special_pattern = (re.compile("|".join(re.escape(token) for token in
                                sorted(self._special, key=len, reverse=True)))
                                 if self._special else None)
        self._pieces = tuple(self._token_bytes(token, kind)
                             for token, kind in zip(tokens, types))
        self._cached_bpe = lru_cache(maxsize=32768)(self._bpe)

    @staticmethod
    def _token_bytes(token: str, kind: int) -> bytes:
        if kind in (2, 3, 4):
            return token.encode("utf-8")
        if kind == 5:
            return b""
        if kind == 6:
            return bytes([int(token[3:-1], 16)])
        raw = bytearray()
        for char in token:
            byte = _CHAR_TO_BYTE.get(char)
            if byte is not None:
                raw.append(byte)
            else:
                # The checkpoint also contains non-byte-alphabet spellings;
                # llama.cpp exposes these as diagnostics, not ordinary UTF-8.
                marker = f"[UNK_BYTE_0x{char.encode('utf-8').hex()}{token}]"
                raw.extend(marker.encode("utf-8"))
        return bytes(raw)

    def _bpe(self, fragment: str) -> tuple[int, ...]:
        """Merge lowest ranks first; ties go to the leftmost live pair."""
        symbols = [_BYTE_TO_CHAR[byte] for byte in fragment.encode("utf-8")]
        if not symbols:
            return ()
        before = [i - 1 for i in range(len(symbols))]
        after = [i + 1 for i in range(len(symbols))]
        after[-1] = -1
        queue = []

        def offer(left: int):
            if left < 0:
                return
            right = after[left]
            if right < 0:
                return
            pair = (symbols[left], symbols[right])
            rank = self._ranks.get(pair)
            if rank is not None:
                heappush(queue, (rank, left, right, pair))

        for left in range(len(symbols) - 1):
            offer(left)
        while queue:
            _rank, left, right, pair = heappop(queue)
            if after[left] != right or (symbols[left], symbols[right]) != pair:
                continue
            symbols[left] += symbols[right]
            symbols[right] = ""
            after[left] = after[right]
            if after[right] >= 0:
                before[after[right]] = left
            after[right] = -1
            offer(before[left])
            offer(left)
        return tuple(self._vocab[symbol] for symbol in symbols if symbol)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode literal special tokens too; `add_special` controls BOS/EOS."""
        ids = [self.bos_id] if add_special and self.add_bos else []

        def ordinary(fragment: str):
            for piece in _command_r_fragments(fragment):
                ids.extend(self._cached_bpe(piece))

        at = 0
        if self._special_pattern is not None:
            for match in self._special_pattern.finditer(text):
                ordinary(text[at:match.start()])
                ids.append(self._special[match.group()])
                at = match.end()
        ordinary(text[at:])
        if add_special and self.add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids) -> str:
        """Join token bytes before UTF-8 decoding; render special spellings."""
        pieces = []
        for token_id in ids:
            if not isinstance(token_id, int) or not 0 <= token_id < len(self._pieces):
                raise ValueError(f"invalid token ID: {token_id!r}")
            pieces.append(self._pieces[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")
