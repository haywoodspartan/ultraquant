"""Train a deployed library's router from a local LM Studio model.

This is :mod:`ultraquant.forge.distill` pointed at a real session rather than at
a gate's synthetic categories. It walks the router's categories, asks the loaded
model how someone might phrase a query about each, filters to content tokens,
teaches them, and writes the router back.

**It holds itself to the measured ceiling.** §11.13 found that this mechanism
does not scale: at 12 phrasings per category the decoy control holds at 1.000,
and at 30 the margin looks far better while a quarter of unrelated queries get
misrouted. So :data:`~ultraquant.forge.distill.SAFE_PHRASINGS_PER_CATEGORY` is
a cap here, not a default to raise — and this module refuses to exceed it rather
than exposing a knob that silently invalidates the result.

**Synthetic categories are skipped.** A forged library carries families named
``family_007`` whose keywords are ``sym``, ``0``, ``1``. Asking a model for
"different ways to ask about family_007" produces confident nonsense that would
then be taught to the router as though it meant something. Only categories with
real vocabulary are trained, and the rest are reported as skipped rather than
quietly dropped.

**A held-out split is measured, not assumed.** Half the generated phrasings are
taught and half withheld, and routing accuracy on the withheld half is reported
before and after — together with a decoy check, because the failure mode this
mechanism has is becoming more willing to answer rather than better informed.

Run it::

    python -m ultraquant.forge.train_from_llm uq_home --model <id>
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.forge.distill import (
    SAFE_PHRASINGS_PER_CATEGORY,
    content_tokens,
    distil_into_router,
    generate_phrasings,
)
from ultraquant.interpreter.llmls import LMStudioUnavailable, TeacherPanel

__all__ = ["TrainingReport", "train_router", "main"]

#: Categories a forged library invents for synthetic families. Their keywords
#: carry no meaning, so a model asked about them will invent some.
_SYNTHETIC = re.compile(r"^(family|cat|category|class)[_-]?\d+$", re.I)

#: Keywords too generic to describe a topic to a model.
_UNINFORMATIVE = frozenset({"sym", "family", "0", "1", "2", "what", "where",
                            "who", "fact"})

#: Queries that belong to none of the trained categories. Routing them into one
#: is the failure this mechanism actually has.
_DECOYS = [
    "how do I fix a leaking tap",
    "what time does the train leave",
    "recipe for banana bread",
    "the weather forecast for tomorrow",
    "who won the football match last night",
]


@dataclass
class TrainingReport:
    """What a training run did and what it cost.

    Attributes:
        trained: Category -> how many phrasings were taught.
        skipped: Category -> why it was not trained.
        before: Routing accuracy on held-out phrasings, pre-training.
        after: The same, post-training.
        decoys_before: Fraction of decoys correctly left alone, pre-training.
        decoys_after: The same, after. A fall here is the failure mode.
        model: Which teacher was used.
    """

    trained: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    before: float = 0.0
    after: float = 0.0
    decoys_before: float = 0.0
    decoys_after: float = 0.0
    model: str = ""

    @property
    def phrasings(self) -> int:
        """Total phrasings taught."""
        return sum(self.trained.values())

    def as_text(self) -> str:
        """A printable report, with the decoy check next to the win."""
        lines = [f"teacher: {self.model}",
                 f"trained {len(self.trained)} categor(y/ies), "
                 f"{self.phrasings} phrasing(s)"]
        for name, count in sorted(self.trained.items()):
            lines.append(f"    {name:<14} {count} phrasing(s)")
        for name, why in sorted(self.skipped.items()):
            lines.append(f"    {name:<14} skipped: {why}")
        lines.append("")
        lines.append(f"held-out routing   {self.before:.3f} -> {self.after:.3f}"
                     f"   ({self.after - self.before:+.3f})")
        lines.append(f"decoys left alone  {self.decoys_before:.3f} -> "
                     f"{self.decoys_after:.3f}"
                     + ("   <- FELL: the router became more willing to answer, "
                        "not better informed"
                        if self.decoys_after < self.decoys_before else "   (held)"))
        return "\n".join(lines)


def _topic_for(category: str, keywords: set) -> str | None:
    """A human-readable description of a category, or None if it has none.

    Built from the category's own keywords rather than a hand-written table, so
    a library with categories this module has never heard of still trains. A
    category whose vocabulary is all placeholders returns None and is skipped —
    the model would otherwise invent a topic and the router would learn it.
    """
    useful = sorted(w for w in keywords
                    if w not in _UNINFORMATIVE and len(w) > 2
                    and not w.isdigit() and not _SYNTHETIC.match(w))
    if len(useful) < 2:
        return None
    return f"{category} ({', '.join(useful[:6])})"


def _score(router, cases: list, top_k: int = 1) -> float:
    """Fraction of ``(query, category)`` pairs routed correctly."""
    if not cases:
        return 0.0
    hit = 0
    for query, truth in cases:
        ranked = router.route(query, top_k=top_k)
        if ranked and ranked[0][0] == truth:
            hit += 1
    return hit / len(cases)


def _decoys_left_alone(router, categories: set) -> float:
    """Fraction of unrelated queries the router does *not* claim."""
    missed = 0
    for query in _DECOYS:
        ranked = router.route(query, top_k=1)
        if not ranked or ranked[0][0] not in categories:
            missed += 1
    return missed / len(_DECOYS)


def train_router(session, panel: TeacherPanel, holdout: float = 0.5,
                 seed: int = 0, count: int | None = None) -> TrainingReport:
    """Distil phrasings for every nameable category and teach them.

    Args:
        session: A built session, whose ``router`` is trained in place.
        panel: The teachers.
        holdout: Fraction of phrasings withheld from teaching and used to
            measure. Half by default, so the reported figure is on text the
            router has never seen.
        seed: Split seed.
        count: Phrasings per category. Capped at
            :data:`SAFE_PHRASINGS_PER_CATEGORY` — above it the decoy control
            breaks, so this refuses to go higher rather than offering a knob
            that invalidates the measurement.

    Returns:
        The :class:`TrainingReport`. The router is saved by the caller.
    """
    router = session.router
    requested = SAFE_PHRASINGS_PER_CATEGORY if count is None else int(count)
    per_category = min(requested, SAFE_PHRASINGS_PER_CATEGORY)

    report = TrainingReport(model=", ".join(card.id for card in panel.cards))
    categories = set(router._base)
    generated: dict[str, list[str]] = {}

    for category in sorted(categories):
        if _SYNTHETIC.match(category):
            report.skipped[category] = "synthetic family, no real topic"
            continue
        topic = _topic_for(category, router._base.get(category, set()))
        if topic is None:
            report.skipped[category] = "no descriptive keywords to ask about"
            continue
        try:
            produced = generate_phrasings(panel, category, topic,
                                          count=per_category)
        except LMStudioUnavailable as exc:
            report.skipped[category] = f"teacher unavailable: {exc}"
            continue
        usable = [p for p in produced.phrasings if content_tokens(p)]
        if len(usable) < 2:
            report.skipped[category] = (
                f"only {len(usable)} usable phrasing(s) came back")
            continue
        generated[category] = usable

    if not generated:
        return report

    rng = random.Random(seed)
    teach: dict[str, list[str]] = {}
    held: list[tuple[str, str]] = []
    for category, phrasings in generated.items():
        shuffled = list(phrasings)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * (1.0 - holdout)))
        teach[category] = shuffled[:cut]
        held += [(p, category) for p in shuffled[cut:]]

    report.before = _score(router, held)
    report.decoys_before = _decoys_left_alone(router, categories)
    for category, phrasings in teach.items():
        report.trained[category] = distil_into_router(router, category,
                                                      phrasings)
    report.after = _score(router, held)
    report.decoys_after = _decoys_left_alone(router, categories)
    return report


def main(argv: list[str] | None = None) -> int:
    """Train a session's router from a local model.

    Returns:
        0 on success, 1 when no teacher is reachable.
    """
    import argparse
    import sys

    from ultraquant.interpreter.thoughts import build_session

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", nargs="?", default="uq_home",
                        help="session directory to train")
    parser.add_argument("--model", action="append", default=None,
                        help="teacher model id; repeatable. Defaults to "
                             "whatever is loaded in LM Studio.")
    parser.add_argument("--count", type=int, default=None,
                        help=f"phrasings per category (capped at "
                             f"{SAFE_PHRASINGS_PER_CATEGORY})")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure against a COPY of the home, changing "
                             "nothing (learning reinforces the vault, so "
                             "skipping the save alone is not enough)")
    args = parser.parse_args(argv)

    try:
        models = args.model
        if not models:
            from ultraquant.interpreter.llmls import catalogue

            models = [card.id for card in catalogue()
                      if card.is_chat and card.loaded]
            if not models:
                print("No chat model is loaded in LM Studio. Load one, or "
                      "pass --model.", file=sys.stderr)
                return 1
        panel = TeacherPanel(models)
    except LMStudioUnavailable as exc:
        print(f"LM Studio unavailable: {exc}", file=sys.stderr)
        return 1

    home = Path(args.home)
    scratch = None
    if args.dry_run:
        # A dry run has to work on a *copy*, not merely skip router.save().
        # CategoryRouter.learn() also calls vault.reinforce(), which writes the
        # shard catalog immediately - so the first "dry" run here really did
        # train the library, and the next run started from the moved baseline
        # (0.109 rather than a clean one). Not saving was not enough to be dry.
        import shutil
        import tempfile

        scratch = Path(tempfile.mkdtemp(prefix="uq_dry_"))
        home = scratch / home.name
        shutil.copytree(Path(args.home), home)

    try:
        session = build_session(home, seed=0)
        report = train_router(session, panel, count=args.count)
        print(report.as_text())
        if args.dry_run:
            print(f"\ndry run - worked on a copy; nothing in "
                  f"{Path(args.home)} changed")
        elif report.trained:
            session.router.save()
            print(f"\nrouter saved to {session.router.path}")
    finally:
        if scratch is not None:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
