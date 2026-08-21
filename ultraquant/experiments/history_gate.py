"""Belief history, askable — and never invented. The history gate.

Every change of mind has been an episode since §11.16's era, a spoken
notice since §11.53, and a queryable provenance since §11.51 — but
"what was the tower material?" answered with the present. The record
was write-only. §11.64 opens the read side: past-tense questions
answer from the revision episodes, every past belief named in order
("It was iron, then steel; it is bronze now, 2 revision(s) on
record"), polarity spoken where the past was a denial — which
required a data-shape fix found in passing: the revision episode
logged bare values, so a §11.48 flip's history would have read "was
steel" when the belief was "not steel"; episodes now log spoken
forms, old episodes keeping their old shape.

The line this gate holds is the §11.51 line one tense back: **history
is recorded, never invented.** A fact stated once has no history, and
the answer says so ("I have no record of it ever being different");
an unheld subject gets a hedge, never a past.

Both arms run the live pipeline; the baseline arm turns
`_HISTORY_ANSWERS` off (past-tense questions answer with the
present), the treatment arm is current. Worlds mix, in world-varying
proportions: once-revised facts, twice-revised facts (order matters),
polarity-flip histories, never-revised facts (the no-record line),
and unheld subjects (the hedge, and the variance's free share).

**The criteria, written before the run.**

1. **Gain** — history correctness (every past value named in order,
   the present marked as present, the revision count right;
   never-revised answering "no record"; unheld hedging) must beat the
   baseline arm by more than the seed-to-seed sd.
2. **History is never invented** — an unheld subject or a
   never-revised fact answered with "It was ..." in the treatment arm
   is a fail, once.
3. **The present is untouched** — present-tense questions answer
   identically in both arms at 1.000.

**It was run twice: the first run scrambled its own worlds, the
second PASSED perfectly.**

The first run failed its present floor at 0.431 in BOTH arms — the
flag could not be the cause, and the cause was the builder's shuffle:
in a history world the statement SEQUENCE is the semantics, and
shuffling ingested half the revisions backwards. The harness lesson
joins §11.58's article keys and §11.63's slot collision: a world must
be built the way the tested thing actually happens. Order preserved,
then:

| arm | history | present |
|---|---:|---:|
| present-only | 0.127 | 1.000 |
| remembers | **1.000** | 1.000 |

**+0.873 at 11.16x seed sd, zero histories invented** — every past
belief named in order with its revision count, polarity flips reading
"It was not steel", never-revised facts answering "no record of it
ever being different", unheld subjects hedging, and the present
identical in both arms. The belief store now answers in three tenses:
what is (recall and derivation), what was (this unit), and why
(§11.51) — all from one record, none of it invented.

Run it::

    python -m ultraquant.experiments.history_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HistoryReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]


@dataclass
class HistoryReport:
    """The two arms, the invention line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        history: Arm -> history-question correctness over all worlds.
        invented: "It was" answers over no-history subjects in
            treatment (must be 0).
        present: Arm -> present-tense correctness (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    history: dict = field(default_factory=dict)
    invented: int = 0
    present: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'history':>8} {'present':>8}"]
        for arm in ("present-only", "remembers"):
            lines.append(f"{arm:<12} {self.history[arm]:>8.3f} "
                         f"{self.present[arm]:>8.3f}")
        lines += [
            f"histories invented (treatment): {self.invented}",
            f"delta (remembers vs present-only) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One history world.

    Returns ``(statements, questions, present_qs)`` where questions
    are ``(question, kind, payload)`` — kind in one/two/flip/none/
    unheld; payload what a correct answer must contain, in order.
    """
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    questions = []
    present_qs = []

    # Once-revised.
    for entity in names[:1 + rng.randrange(1, 3)]:
        first, second = rng.sample(_MATERIALS, 2)
        statements.append(f"the {entity} material is {first}")
        statements.append(f"the {entity} material is {second}")
        questions.append(
            (f"what was the {entity} material?", "one",
             [f"It was {first}", f"it is {second} now",
              "1 revision(s)"]))
        present_qs.append((f"what is the {entity} material?", second))
    # Twice-revised: order matters.
    if rng.random() < 0.8:
        entity = names[3]
        a, b, c = rng.sample(_MATERIALS, 3)
        for material in (a, b, c):
            statements.append(f"the {entity} material is {material}")
        questions.append(
            (f"what was the {entity} material?", "two",
             [f"It was {a}, then {b}", f"it is {c} now",
              "2 revision(s)"]))
    # A polarity flip: the past was a denial.
    if rng.random() < 0.8:
        entity = names[4]
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is not {material}")
        statements.append(f"the {entity} material is {material}")
        questions.append(
            (f"what was the {entity} material?", "flip",
             [f"It was not {material}", f"it is {material} now"]))
    # Never revised: the no-record line.
    entity = names[5]
    material = rng.choice(_MATERIALS)
    statements.append(f"the {entity} material is {material}")
    questions.append(
        (f"what was the {entity} material?", "none",
         [f"I have no record of {entity} material ever being "
          "different", material]))
    # Unheld: the hedge.
    if rng.random() < 0.8:
        questions.append(
            (f"what was the {names[6]} material?", "unheld", []))

    # NO shuffle, deliberately: in a history world the statement
    # sequence IS the semantics - the first run shuffled and scrambled
    # half its own revisions, failing the present floor in both arms
    # identically. A harness must respect the thing it is testing.
    return statements, questions, present_qs


def run_gate(seeds: int = 12) -> HistoryReport:
    """Run both arms over generated history worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`HistoryReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    history_scores = {"present-only": [], "remembers": []}
    present_scores = {"present-only": [], "remembers": []}
    invented = 0

    real_flag = thoughts_module._HISTORY_ANSWERS
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, questions, present_qs = _build_world(rng)

        for arm in ("present-only", "remembers"):
            thoughts_module._HISTORY_ANSWERS = (arm == "remembers")
            root = Path(tempfile.mkdtemp(prefix=f"uq_hist_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind, payload in questions:
                    response, _trace = run_pipeline(question, session)
                    told_past = response.startswith("It was")
                    if kind in ("one", "two", "flip"):
                        won.append(float(
                            told_past
                            and all(p in response for p in payload)))
                    elif kind == "none":
                        won.append(float(
                            all(p in response for p in payload)
                            and not told_past))
                        if arm == "remembers" and told_past:
                            invented += 1
                    else:
                        won.append(float(
                            not told_past
                            and "no record" not in response))
                        if arm == "remembers" and told_past:
                            invented += 1
                history_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, expected in present_qs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                present_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._HISTORY_ANSWERS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = HistoryReport(
        history={arm: statistics.fmean(vals)
                 for arm, vals in history_scores.items()},
        present={arm: statistics.fmean(vals)
                 for arm, vals in present_scores.items()},
        invented=invented,
    )

    contrasts = [history_scores["remembers"][i]
                 - history_scores["present-only"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.invented > 0:
        report.reason = (
            f"a past was invented {report.invented} time(s) - history "
            "is recorded, never invented")
        return report
    if min(report.present.values()) < 1.0:
        report.reason = "present-tense answers regressed"
        return report
    if report.delta <= 0:
        report.reason = f"remembering buys nothing ({report.delta:+.3f})"
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
        f"history questions answer at {report.history['remembers']:.3f} "
        f"against the present-only arm's "
        f"{report.history['present-only']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), zero histories "
        "invented, and the present untouched at 1.000 in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
