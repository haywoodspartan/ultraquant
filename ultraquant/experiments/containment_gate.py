"""Containment corroboration against manufactured consensus. The gate.

§11.40's first library-training run scored an honest zero: four voices
gave four DIFFERENT true elaborations — "system communication using
symbols", "communication", "system communication" — and exact matching
refused to equate them. Three of those four contain one another. The
question this gate decides is whether crediting that containment is
orthography-class (a position and its elaboration are one position) or
meaning-class (the manufactured-consensus error the module's own
docstring forbids).

The mechanism on trial (:func:`~ultraquant.interpreter.llmls.
_containment_core`): a core is an EXISTING position whose tokens are a
subset of its backers' tokens; no word is ever equated with a different
word; a backing position containing any negation token backs nothing;
and the credited answer is the core — the LEAST claim all merged voices
back. The gate drives the REAL ``TeacherPanel._tally`` with fabricated
answer sets over four independent-lineage cards; no LM Studio involved,
because the question is about the accounting, not the models.

**The criteria, written before the run.**

1. **Recall** — on elaboration worlds (a shared kernel plus per-voice
   elaborations), the containment arm must corroborate more often than
   the exact arm by more than the seed-to-seed sd.
2. **Negation control** — worlds where the only containment path runs
   through a negated superset ("<kernel> is a hoax") must corroborate
   in NEITHER arm. One fall is a fail, sd or no sd: this world is why
   the module said "never credited" in the first place.
3. **Paraphrase control** — worlds where voices say the same thing in
   DIFFERENT words must corroborate in neither arm. Containment must
   not become synonym-guessing through the back door.
4. **Exact behavior unchanged** — exact-agreement worlds corroborate
   identically in both arms.

**It was run, and it PASSED — narrowly on margin, absolutely on the
controls.**

| arm | recall | negation falls | paraphrase falls |
|---|---:|---:|---:|
| exact | 0.000 | 0.000 | 0.000 |
| containment | **0.600** | 0.000 | 0.000 |

**+0.600 at 1.16x seed sd** — §11.31's narrow-margin class, said so in
the same tradition — with **zero negation falls, zero paraphrase
falls**, and exact-agreement behavior identical between arms. The
0.400 miss is the overlap-without-nesting worlds ("system
communication" beside "system altered idea": tokens shared, neither a
subset), included at ~35% because they are the live §11.40 run's own
shape and the mechanism's honest ceiling — containment credits true
subsets and nothing looser, which is precisely what keeps it
orthography-class. The "never credited" line in the module docstring
is hereby amended to "credited only through this gate's rules", and
the CONTAINMENT flag that turns it off again is one constant.

Run it::

    python -m ultraquant.experiments.containment_gate
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

__all__ = ["ContainmentReport", "run_gate"]

_KERNELS = (
    ("system communication", "using symbols"),
    ("earth round", "and orbits the sun"),
    ("numbers operations", "on quantities"),
    ("stored instructions", "for a computer"),
    ("planet third", "from the sun"),
)
_PARAPHRASES = (
    ("fast", "quick"), ("large", "big"), ("humid", "damp"),
)
_NEGATION_WRAPS = ("{} is a hoax", "{} is not true", "{} was debunked",
                   "never {}")


@dataclass
class ContainmentReport:
    """The two arms across four world types, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        recall: Arm -> corroboration rate on elaboration worlds.
        negation_falls: Arm -> corroborations on negation worlds.
        paraphrase_falls: Arm -> corroborations on paraphrase worlds.
        exact_match: True when exact worlds behaved identically.
        delta: Recall contrast (containment minus exact).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    recall: dict = field(default_factory=dict)
    negation_falls: dict = field(default_factory=dict)
    paraphrase_falls: dict = field(default_factory=dict)
    exact_match: bool = True
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<12} {'recall':>7} {'negation falls':>15} "
                 f"{'paraphrase falls':>17}"]
        for arm in ("exact", "containment"):
            lines.append(f"{arm:<12} {self.recall[arm]:>7.3f} "
                         f"{self.negation_falls[arm]:>15.3f} "
                         f"{self.paraphrase_falls[arm]:>17.3f}")
        lines += [
            f"exact-agreement worlds identical: {self.exact_match}",
            f"delta (recall, containment vs exact) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _panel_and_answers(replies: dict):
    """A real TeacherPanel over four independent cards, with canned text."""
    from ultraquant.interpreter.llmls import Answer, ModelCard, TeacherPanel

    cards = [
        ModelCard("alpha-7b", "alphanet", "alphaco"),
        ModelCard("beta-9b", "betanet", "betaco"),
        ModelCard("gamma-13b", "gammanet", "gammaco"),
        ModelCard("delta-20b", "deltanet", "deltaco"),
    ]
    panel = TeacherPanel.__new__(TeacherPanel)
    panel.cards = cards
    answers = {card.id: Answer(prompt="what is the subject?",
                               text=replies.get(card.id, ""),
                               model=card.id)
               for card in cards if replies.get(card.id)}
    return panel, answers


def _tally(replies: dict) -> bool:
    """Whether the real _tally corroborates this canned answer set."""
    panel, answers = _panel_and_answers(replies)
    consensus = panel._tally("what is the subject?", answers, {})
    return consensus.corroborated


def run_gate(seeds: int = 10) -> ContainmentReport:
    """Run both arms over generated answer-set worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ContainmentReport`.
    """
    from ultraquant.interpreter import llmls

    recall = {"exact": [], "containment": []}
    negation = {"exact": 0, "containment": 0}
    paraphrase = {"exact": 0, "containment": 0}
    exact_same = True

    real_flag = llmls.CONTAINMENT
    try:
        for seed in range(seeds):
            rng = random.Random(seed)
            kernel, elaboration = rng.choice(_KERNELS)
            word_a, word_b = rng.choice(_PARAPHRASES)
            wrap = rng.choice(_NEGATION_WRAPS)

            # ~35% of worlds are overlap-without-nesting - the live run's
            # "evolves through usage" beside "system communication": tokens
            # shared, neither a subset. Containment credits true subsets
            # only, so BOTH arms honestly miss, and those worlds carry the
            # contrast's variance and the mechanism's recorded ceiling.
            head, tail = kernel.split()
            if rng.random() < 0.35:
                elaboration_world = {
                    "alpha-7b": f"{head} {tail}",
                    "beta-9b": f"{head} altered {rng.choice(['form', 'idea'])}",
                    "gamma-13b": "entirely unrelated notion",
                    "delta-20b": "another different reply",
                }
            else:
                elaboration_world = {
                    "alpha-7b": kernel,
                    "beta-9b": f"{kernel} {elaboration}",
                    "gamma-13b": "entirely unrelated notion",
                    "delta-20b": "another different reply",
                }
            worlds = {
                "elaboration": elaboration_world,
                # The only containment path runs through a negation.
                "negation": {
                    "alpha-7b": kernel,
                    "beta-9b": wrap.format(kernel),
                    "gamma-13b": "entirely unrelated notion",
                },
                # Same meaning, different words: must never corroborate.
                "paraphrase": {
                    "alpha-7b": f"it is {word_a}",
                    "beta-9b": f"it is {word_b}",
                },
                # Exact agreement: both arms corroborate identically.
                "exact": {
                    "alpha-7b": kernel,
                    "beta-9b": kernel,
                },
            }

            for arm, flag in (("exact", False), ("containment", True)):
                llmls.CONTAINMENT = flag
                recall[arm].append(float(_tally(worlds["elaboration"])))
                negation[arm] += int(_tally(worlds["negation"]))
                paraphrase[arm] += int(_tally(worlds["paraphrase"]))
            llmls.CONTAINMENT = False
            exact_off = _tally(worlds["exact"])
            llmls.CONTAINMENT = True
            exact_on = _tally(worlds["exact"])
            if exact_off != exact_on:
                exact_same = False
    finally:
        llmls.CONTAINMENT = real_flag

    report = ContainmentReport(
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall.items()},
        negation_falls={arm: negation[arm] / seeds
                        for arm in ("exact", "containment")},
        paraphrase_falls={arm: paraphrase[arm] / seeds
                          for arm in ("exact", "containment")},
        exact_match=exact_same,
    )
    contrasts = [recall["containment"][i] - recall["exact"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if max(report.negation_falls.values()) > 0:
        report.reason = (
            "the negation control failed: agreement was read through a "
            "denial - the exact error the module's 'never credited' line "
            "existed to prevent")
        return report
    if max(report.paraphrase_falls.values()) > 0:
        report.reason = (
            "the paraphrase control failed: different words corroborated "
            "- containment became synonym-guessing through the back door")
        return report
    if not report.exact_match:
        report.reason = "exact-agreement behavior changed between arms"
        return report
    if report.delta <= 0:
        report.reason = f"containment credits nothing ({report.delta:+.3f})"
        return report
    if report.seed_sd <= 0.0:
        report.reason = (f"every world gave the same contrast "
                         f"({report.delta:+.3f}); nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {report.delta:+.3f} is "
                         f"{report.margin_ratio:.2f}x the seed sd "
                         f"({report.seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"containment lifts elaboration-world corroboration "
        f"{report.recall['exact']:.3f} -> "
        f"{report.recall['containment']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with zero negation falls, "
        "zero paraphrase falls, and exact behavior unchanged")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
