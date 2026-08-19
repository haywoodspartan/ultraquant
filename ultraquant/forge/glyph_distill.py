"""Distil glyph *variants* from a local model, to train the pattern experts.

The routing distillation (§11.13) taught the router how people phrase queries.
This is the same idea pointed at the experts themselves: the library's glyph
classifiers have only ever seen the forge's synthetic noise around eight
canonical patterns, and a local model can draw **variants** — a plus shifted a
cell left, a thick-armed plus — that widen each family beyond what pixel noise
reaches.

**Which model can draw is the opposite of which model can phrase.** Probed
before building: the 7B that carried the routing distillation echoed the
canonical twice and mislabeled a vertical line as a plus — zero usable from
three. The 31B produced a shifted plus and a thick plus, two usable of three.
Smallest-first was measured right for text generation and measured wrong here;
neither ordering is a principle, both are task measurements.

**A generated glyph is training data only after three checks**, because a
mislabeled variant poisons the expert it claims to train:

1. **Parse** — exactly five lines of exactly five ``#``/``.`` characters, after
   stripping prose and template tokens. Anything else is not a glyph.
2. **Novelty** — byte-identical echoes of the canonical (the commonest failure)
   and duplicates of already-accepted variants are dropped.
3. **No contradiction** — the variant must be *nearest to its own label's*
   canonical among all canonicals, by pixel agreement. The 7B's "plus" that was
   actually a line dies here. A low-similarity variant of the right family
   passes — novelty is the point — but one that looks more like a *different*
   label than its own is exactly the poison the check exists for.

The check deliberately uses canonical patterns rather than the trained
recognizer: filtering by what the current expert already accepts would bias the
data toward what it already knows, and the variants it cannot yet recognise are
the valuable ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ultraquant.pattern.recognition import PATTERNS, render

__all__ = ["VariantReport", "parse_glyphs", "contradiction", "generate_variants"]

_ROW = re.compile(r"^[#.]{5}$")
_TEMPLATE_TOKEN = re.compile(r"<\|[^|>]*\|>")


@dataclass
class VariantReport:
    """What one label's generation produced.

    Attributes:
        label: The glyph label the variants claim.
        accepted: Variants that passed every check, as row-lists.
        rejected: ``(reason, rows-or-text)`` for everything dropped.
        model: The teacher.
    """

    label: str
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    model: str = ""


def parse_glyphs(text: str) -> list[list[str]]:
    """Every well-formed 5x5 glyph in a completion, in order.

    Models wrap glyphs in prose, number them, and leak template tokens; rows
    are collected wherever five valid lines appear consecutively.
    """
    lines = [_TEMPLATE_TOKEN.sub("", line).strip()
             for line in text.splitlines()]
    glyphs: list[list[str]] = []
    run: list[str] = []
    for line in lines:
        if _ROW.match(line):
            run.append(line)
            if len(run) == 5:
                glyphs.append(run)
                run = []
        else:
            run = []
    return glyphs


def _shift(rows: list[str], dx: int, dy: int) -> list[str]:
    """Translate a glyph, padding with dots; content shifted off is lost."""
    blank = "." * 5
    grid = [blank] * 5
    for y, row in enumerate(rows):
        ny = y + dy
        if not 0 <= ny < 5:
            continue
        shifted = ("." * max(0, dx) + row + "." * max(0, -dx))
        shifted = shifted[max(0, -dx):][:5].ljust(5, ".")
        grid[ny] = shifted
    return grid


def _agreement(rows_a: list[str], rows_b: list[str]) -> float:
    """Best cell agreement between two glyphs over small translations.

    Positional agreement alone is the wrong metric for families whose identity
    survives translation, and it failed measurably: horizontal bars drawn one
    row lower than the canonical - unambiguously stripes to a reader - scored
    **0.000** against ``stripes_h`` (every cell anti-aligned) and 0.4 against
    ``plus``, so a correct variant was rejected as mislabeled. Taking the best
    agreement over shifts of up to two cells in each axis scores that variant
    1.000 against its own family while ``plus`` under any shift stays far
    lower.

    Jaccard over the *set* pixels, not raw cell agreement: sparse glyphs agree
    on empty cells almost everywhere, so dot-dominated scoring called nearly
    everything plus-adjacent. The remaining honest limit: a bare 5-cell
    vertical segment labeled "plus" passes, because among these eight anchors
    it genuinely is nearest the plus arm - there is no "single line" family
    for it to contradict into. A 25-cell grid with eight anchors cannot
    adjudicate shapes outside its vocabulary.
    """
    b = render(rows_b)
    b_set = {i for i, v in enumerate(b) if v}
    best = 0.0
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            a = render(_shift(rows_a, dx, dy))
            a_set = {i for i, v in enumerate(a) if v}
            union = a_set | b_set
            if union:
                best = max(best, len(a_set & b_set) / len(union))
    return best


def contradiction(rows: list[str], label: str) -> str | None:
    """The label this variant resembles MORE than its own, or None.

    Judged against the canonical patterns, not the trained recognizer — the
    recognizer would bias acceptance toward what it already knows, and the
    variants it cannot yet place are the valuable ones. A tie goes to the
    claimed label: equal resemblance is not evidence of mislabeling.
    """
    own = _agreement(rows, PATTERNS[label])
    for other, canonical in PATTERNS.items():
        if other == label:
            continue
        if _agreement(rows, canonical) > own:
            return other
    return None


def strict_contradiction(rows: list[str], label: str,
                         margin: float = 0.0,
                         floor: float = 0.0) -> str | None:
    """:func:`contradiction` with the two laxities the blind reader exposed.

    ARCHITECTURE §11.28 measured the plain rule admitting 26% of the banks
    against an independent reading, and the rejections decompose exactly onto
    the two checks this adds:

    * **floor** — a glyph must resemble its own canonical at least this much.
      18 of the 31 reader-rejected variants read as *nothing*: they were each
      nearest their claimed label, but weakly, and "nearest" was the whole
      old test.
    * **margin** — the own-label agreement must beat the best other label by
      at least this much. The other 13 rejections read as a *different*
      label; most had won their claim by a whisker.

    Returns the offending other label, the sentinel ``"(ambiguous)"`` for a
    floor failure, or None when the variant passes. With both parameters at
    0.0 this is exactly the historical rule.

    **The gate ran, and no operating point earned a caller** — see
    ``experiments/validator_gate.py``: held-half recall 0.050 against a
    0.50 bar, unstable calibration across splits. The reader rejects on
    shape gestalt, which agreement thresholds do not encode, so this
    function is the built-but-not-adopted shelf's third resident, beside
    :func:`~ultraquant.pattern.recognition.center_glyph` and
    :func:`~ultraquant.pattern.recognition.scale_normalize`. Bank
    admission goes through a blind reading (§11.28), permanently.
    """
    own = _agreement(rows, PATTERNS[label])
    if own < floor:
        return "(ambiguous)"
    best_other, offender = 0.0, None
    for other, canonical in PATTERNS.items():
        if other == label:
            continue
        score = _agreement(rows, canonical)
        if score > best_other:
            best_other, offender = score, other
    if best_other > own or own - best_other < margin:
        return offender
    return None


def generate_variants(client, model: str, label: str, count: int = 6,
                      max_tokens: int = 700) -> VariantReport:
    """Ask a model to draw variants of one canonical glyph, validated.

    Args:
        client: An :class:`~ultraquant.interpreter.lmstudio.LMStudioClient`.
        model: Teacher model id.
        label: One of :data:`PATTERNS`' labels.
        count: Variants to request. Expect roughly half to survive validation.
        max_tokens: Generous; a glyph is ~36 tokens but models pad with prose.

    Returns:
        The :class:`VariantReport`.
    """
    from ultraquant.interpreter.llmls import NO_REASONING

    canonical = PATTERNS[label]
    newline = chr(10)
    prompt = (
        f"Here is a 5x5 glyph of '{label}', drawn with # and . characters:"
        f"{newline}{newline}{newline.join(canonical)}{newline}{newline}"
        f"Draw {count} DIFFERENT variants of '{label}' on the same 5x5 grid - "
        "for example shifted by one cell, thicker, or thinner. Every variant "
        f"must still read as '{label}'. Each variant: exactly five lines, each "
        "line exactly five characters, only # and . characters. Separate "
        "variants with a blank line. Output nothing else."
    )
    report = VariantReport(label=label, model=model)
    answer = client.complete(prompt, model=model, system="",
                             max_tokens=max_tokens,
                             reasoning_effort=NO_REASONING)
    seen = {tuple(canonical)}
    for rows in parse_glyphs(answer.text or ""):
        key = tuple(rows)
        if key in seen:
            report.rejected.append(("echo or duplicate", rows))
            continue
        clash = contradiction(rows, label)
        if clash is not None:
            report.rejected.append((f"resembles {clash} more", rows))
            continue
        seen.add(key)
        report.accepted.append(rows)
    return report
