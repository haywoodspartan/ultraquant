"""Command-R's text becomes exactly llama.cpp's token ids. The tokenizer gate.

The second unit of running the user's own Command-R 08-2024: a model
reads token ids, not text, and an id that differs from the one it was
trained on is a different word. The tokenizer is built only from the
GGUF's own metadata - byte-level BPE, 256,000 tokens, 253,333 merges,
and Command-R's pre-tokenizer, which splits digits one per token before
the GPT-2 pattern ever sees them.

**The oracle is llama.cpp itself**: `llama-server.exe` from LM Studio's
runtime folder, serving this exact GGUF on 127.0.0.1, answering
/tokenize and /detokenize. GPT-6 Astra wrote the tokenizer; Claude wrote
this exam.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 c4361c51...) before the tokenizer existed:

1. **Identical ids** on a battery of 200+ texts across every category
   the pre-registration names, plus 40 sentences from this repo's own
   ARCHITECTURE.md and JOURNAL.md: 0 mismatched sequences.
2. **The exam can fail**: the same battery through the tokenizer with
   ONE planted defect - digits not pre-split one per token - must
   mismatch at least one text, or the harness is void.
3. **Lossless both ways**: decode(encode(t)) == t for every text, and
   our decode(ids) equals the oracle's /detokenize(ids).
4. **Nothing else moves**, and **the full suite is green** - checked
   outside this module.

**The first run was VOID, and the reason is a finding.** The planted
defect - digits not pre-split - mismatched 0 of 207 texts, because it is
not a defect for this model: 0 of Command-R's 253,333 merges touch a
digit, so BPE can never join a digit to anything and the pre-split is
behaviourally redundant. No text could reveal it. Amendment A (sha256
1ea83dd3...), made after that run and before the second, planted a
defect of a different kind chosen a priori: GPT-2's whitespace
lookahead removed, so the last space of a run no longer attaches to the
next word as a "Gword" token.

**PASSED on all three measured criteria, with the harness shown able to
fail**: 207 texts - prose, numbers, whitespace, contractions, seven
scripts, emoji with ZWJ and skin tones, code, URLs, and 40 sentences of
this repo's own prose - produce exactly llama.cpp's ids, 0 mismatched;
decode(encode(t)) == t for all 207; our decode equals llama.cpp's
/detokenize on all 207. The whitespace defect mismatched 15 texts. The
real tokenizer's numbers did not move between the two runs.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ORACLE", "TokenizerReport", "battery", "planted_digits",
           "planted_whitespace", "run_gate"]

ORACLE = "http://127.0.0.1:8089"
COMMAND_R = Path(r"J:\Models\bartowski\c4ai-command-r-08-2024-GGUF"
                 r"\c4ai-command-r-08-2024-Q4_K_S.gguf")
_ROOT = Path(__file__).resolve().parents[2]


def _post(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        ORACLE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def oracle_ids(text: str) -> list:
    return _post("/tokenize", {"content": text, "add_special": True})["tokens"]


def oracle_text(ids: list) -> str:
    return _post("/detokenize", {"tokens": ids})["content"]


def _doc_sentences(count: int = 40) -> list:
    """Sentences from the repo's own prose - evenly spaced, deterministic."""
    found = []
    for name in ("ARCHITECTURE.md", "JOURNAL.md"):
        text = (_ROOT / name).read_text(encoding="utf-8")
        for part in re.split(r"(?<=[.!?])\s+", text):
            part = part.strip()
            if 30 <= len(part) <= 220:
                found.append(part)
    step = max(1, len(found) // count)
    return found[::step][:count]


def battery() -> list:
    """200+ texts: every category the pre-registration names."""
    texts = [
        # prose
        "Hello, world! The tower is 300 meters tall.",
        "The quick brown fox jumps over the lazy dog.",
        "UltraQuant keeps knowledge in a catalogued shard library.",
        "It abstains rather than fabricates.",
        "A sentence with a semicolon; and a colon: here.",
        "Parentheses (like these) and [brackets] and {braces}.",
        "Quotes: \"double\" and 'single' and `backticks`.",
        "Ellipsis... and em-dash - and en-dash too.",
        "ALL CAPS SENTENCE FOR EMPHASIS.",
        "MiXeD CaSe WoRdS",
        # numbers
        "0", "7", "42", "300", "2024", "123456789", "3.14159", "-273.15",
        "1,000,000", "1 000 000", "10^9", "2+2=4", "x=12;y=34", "v2.46.0",
        "Call 555-0199 now", "IPv4 192.168.0.1", "0x1F 0b1010 0o17",
        "Room 101, floor 3", "The year 1969 and 2001", "£20 €15 $9.99 ¥1000",
        "5%", "3/4 cup", "1st 2nd 3rd 4th", "12:30:45", "2026-09-28",
        # punctuation runs
        "!!!", "???", "!?!?", "...", "---", "***", "~~~", "@@@###$$$",
        "a!b@c#d$e%f^g&h*i(j)k", "end.", ".start", "(nested (parens))",
        # whitespace
        " ", "  ", "   leading spaces", "trailing spaces   ", "a  b   c    d",
        "\t", "tab\tseparated\tvalues", "\n", "line one\nline two",
        "para one\n\npara two", "crlf\r\nline", "\n\n\n", " \n \n ",
        "mixed \t \n whitespace", "   leading spaces\tand\ttabs\n",
        # contractions
        "it's", "IT'S", "don't", "DON'T", "they're", "THEY'RE", "we've",
        "WE'VE", "I'm", "I'M", "you'll", "YOU'LL", "he'd", "HE'D",
        "it's what they're saying we've done", "o'clock", "rock 'n' roll",
        # unicode letters
        "naïve café résumé", "Ångström Øresund", "straße", "ﬁnance ligature",
        "東京タワー", "中文分词测试", "日本語のテキスト", "한국어 문장",
        "Привет, мир!", "Съешь же ещё этих мягких французских булок",
        "مرحبا بالعالم", "नमस्ते दुनिया", "Γειά σου Κόσμε", "שלום עולם",
        "Tiếng Việt có dấu", "ไทยภาษา",
        # emoji
        "🙂", "🙂🙂🙂", "I ❤️ this", "👍🏽 skin tone", "👨‍👩‍👧‍👦 family",
        "🏳️‍🌈 flag", "emoji at end 🚀", "🚀 at start", "mixed 🙂 text 🎉 here",
        # code
        "def f(x):\n    return x + 1\n",
        "for (int i = 0; i < n; ++i) { sum += a[i]; }",
        "SELECT * FROM users WHERE id = 42;",
        "const x = () => { return 'y'; };",
        "import numpy as np  # comment",
        "if a == b and c != d:\n\tpass",
        "<div class=\"box\">text</div>",
        "{\"key\": [1, 2, 3], \"ok\": true}",
        "x += 1; y -= 2; z *= 3; w /= 4",
        "#include <stdio.h>",
        # urls and paths
        "https://example.com/path?q=1&r=2#frag",
        "http://127.0.0.1:8089/tokenize",
        "C:\\Users\\name\\file.txt",
        "/usr/local/bin/python3",
        "user@example.com",
        # mixed
        "Command-R 08-2024 has 32B params, 40 blocks, 64 heads.",
        "The 4090 has 24 GB; the model is 17.6 GB.",
        "RoPE θ = 4,000,000 and ε = 1e-5",
        "α β γ δ ε ζ η θ",
        "∑ ∫ √ ∞ ≠ ≤ ≥",
        "\u00a0non-breaking\u00a0space",
        "zero\u200bwidth",
        "",
    ]
    # programmatic: numbers against words and separators
    for n in (1, 12, 123, 1234, 12345):
        for sep in ("", " ", ",", "."):
            texts.append(f"a{sep}{n}{sep}b")
    for word in ("the", "The", "THE", "tower", "Tower"):
        for pad in (" ", "  ", "\n", "\t"):
            texts.append(f"{pad}{word}{pad}")
    texts.extend(_doc_sentences(40))
    return texts


class planted_digits:
    """The FIRST planted defect: digits NOT pre-split one per token.

    Kept because it is the record: it could not fail. Command-R's 253,333
    merges touch no digit, so BPE can never join a digit to anything and
    the pre-split is behaviourally redundant - no text reveals this
    defect, and the first run was void by criterion 2.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special: bool = True) -> list:
        from ultraquant.infer import tokenizer as module

        real = module._command_r_fragments

        def one_pass(chunk: str):
            kinds = [module._kind(char) for char in chunk]
            yield from module._gpt2_fragments(chunk, kinds, 0, len(chunk))

        module._command_r_fragments = one_pass
        try:
            return self.tokenizer.encode(text, add_special=add_special)
        finally:
            module._command_r_fragments = real


class planted_whitespace:
    r"""Amendment A's defect, chosen a priori: no whitespace lookahead.

    GPT-2's \s+(?!\S) leaves the last space of a run to attach to the
    next word as a "Gword" token; plain \s+ swallows it. Those merges
    exist in this vocabulary, so the defect is visible - if the harness
    can see anything.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special: bool = True) -> list:
        from ultraquant.infer import tokenizer as module

        real = module._gpt2_fragments

        def no_lookahead(text_, kinds, start, end):
            pieces = list(real(text_, kinds, start, end))
            merged = []
            # the lone space the lookahead handed to the next word goes
            # back onto the whitespace run before it
            for piece in pieces:
                if (merged and piece[:1] == " " and len(piece) > 1
                        and merged[-1] and merged[-1].isspace()):
                    merged[-1] += " "
                    merged.append(piece[1:])
                else:
                    merged.append(piece)
            yield from merged

        module._gpt2_fragments = no_lookahead
        try:
            return self.tokenizer.encode(text, add_special=add_special)
        finally:
            module._gpt2_fragments = real


@dataclass
class TokenizerReport:
    """Whether our ids are llama.cpp's ids.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        texts: Battery size (200+ required).
        mismatched: Texts whose ids differed from the oracle (need 0).
        planted_mismatched: The same under the planted defect (need > 0).
        round_trip_failures: decode(encode(t)) != t (need 0).
        detok_disagreements: our decode != oracle /detokenize (need 0).
        first_mismatch: The first differing text, with both id lists.
        reason: Plain-language verdict.
    """

    passes: bool
    texts: int = 0
    mismatched: int = 0
    planted_mismatched: int = 0
    round_trip_failures: int = 0
    detok_disagreements: int = 0
    first_mismatch: tuple = field(default_factory=tuple)
    reason: str = ""


def run_gate(tokenizer=None, planted=None) -> TokenizerReport:
    """Every criterion against the live oracle.

    ``planted`` is the same tokenizer with the pre-registered defect
    (digits not pre-split); the caller builds it from the real one.
    """
    from ultraquant.infer.tokenizer import Tokenizer

    tok = tokenizer or Tokenizer.from_gguf(COMMAND_R)
    planted = planted if planted is not None else planted_whitespace(tok)
    report = TokenizerReport(passes=False)
    texts = battery()
    report.texts = len(texts)
    for text in texts:
        want = oracle_ids(text)
        got = tok.encode(text, add_special=True)
        if got != want:
            report.mismatched += 1
            if not report.first_mismatch:
                report.first_mismatch = (text, got[:24], want[:24])
        if planted is not None and planted.encode(text, add_special=True) != want:
            report.planted_mismatched += 1
        body = got[1:] if got[:1] == [5] else got
        if tok.decode(body) != text:
            report.round_trip_failures += 1
        if want and tok.decode(want[1:]) != oracle_text(want[1:]):
            report.detok_disagreements += 1
    valid = planted is not None and report.planted_mismatched > 0
    report.passes = (valid and report.texts >= 200 and report.mismatched == 0
                     and report.round_trip_failures == 0
                     and report.detok_disagreements == 0)
    if not valid:
        report.reason = (f"VOID: the planted defect mismatched "
                         f"{report.planted_mismatched} texts")
    elif report.passes:
        report.reason = (f"PASS: {report.texts} texts, 0 id mismatches, "
                         f"lossless both ways; the planted defect "
                         f"mismatched {report.planted_mismatched}")
    else:
        report.reason = (f"FAIL: {report.mismatched} mismatched, "
                         f"{report.round_trip_failures} round-trip, "
                         f"{report.detok_disagreements} detokenize; first: "
                         f"{report.first_mismatch}")
    return report


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    report = run_gate()
    print(report.reason)
    print(f"  texts {report.texts}; planted defect mismatched "
          f"{report.planted_mismatched}; {time.perf_counter() - started:.1f}s")
