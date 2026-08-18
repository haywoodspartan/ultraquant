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
from ultraquant.interpreter.lmstudio import LMStudioClient

__all__ = ["TrainingReport", "train_router",
           "train_next_voice", "parameter_count", "main"]

#: Categories a forged library invents for synthetic families. Their keywords
#: carry no meaning, so a model asked about them will invent some.
_SYNTHETIC = re.compile(r"^(family|cat|category|class)[_-]?\d+$", re.I)

#: Keywords too generic to describe a topic to a model.
_UNINFORMATIVE = frozenset({"sym", "family", "0", "1", "2", "what", "where",
                            "who", "fact"})

#: Parameter count parsed from a model id ("mistral-7b" -> 7.0, "qwen3-48b"
#: -> 48.0). The catalogue does not report size in bytes, but community naming
#: reliably carries the parameter count, and that is the ordering "smallest to
#: largest" actually means. A model whose id names no size sorts last: its cost
#: is unknown, and unknown does not belong at the front of a cost-ordered
#: queue.
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)


def parameter_count(model_id: str) -> float:
    """Billions of parameters named in a model id, or +inf when unnamed.

    ``max``, not ``min``, when an id names several counts: MoE naming carries
    both totals and actives ("qwen3.6-35b-a3b" is 35B total, 3B active), and
    the total is what decides load cost - the first version took ``min`` and
    ordered a 19.6 GB model in front of a 4 GB one.
    """
    matches = _PARAM_RE.findall(model_id.replace("_", "-"))
    return max(float(m) for m in matches) if matches else float("inf")


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
    #: Category -> the phrasings actually taught (the held half excluded).
    taught: dict = field(default_factory=dict)
    #: True when the decoy gate fired and this run's teaching was undone.
    rolled_back: bool = False

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
        if self.rolled_back:
            lines.append("ROLLED BACK - this run made the router claim "
                         "unrelated queries, so its teaching was undone; "
                         "nothing was kept and no budget was consumed")
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


def _trainable_categories(router, report: TrainingReport) -> dict[str, str]:
    """Category -> topic for everything worth asking a model about."""
    topics: dict[str, str] = {}
    for category in sorted(router._base):
        if _SYNTHETIC.match(category):
            report.skipped[category] = "synthetic family, no real topic"
            continue
        topic = _topic_for(category, router._base.get(category, set()))
        if topic is None:
            report.skipped[category] = "no descriptive keywords to ask about"
            continue
        topics[category] = topic
    return topics


def _teach_and_measure(session, generated: dict, report: TrainingReport,
                       holdout: float, seed: int) -> TrainingReport:
    """Split, teach, and measure - shared by both training modes."""
    router = session.router
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

    categories = set(router._base)
    report.before = _score(router, held)
    report.decoys_before = _decoys_left_alone(router, categories)
    for category, phrasings in teach.items():
        report.trained[category] = distil_into_router(router, category,
                                                      phrasings)
    report.after = _score(router, held)
    report.decoys_after = _decoys_left_alone(router, categories)
    # What was actually taught, for callers that keep a ledger. The held half
    # never touched the router and must not consume absorption budget.
    report.taught = teach

    # The decoy check is a gate, not a caption. Run 4 of the sequential
    # training taught a leaked chat-template token, decoys fell 1.000 -> 0.800,
    # and the report said so while the router was saved anyway - a warning
    # where a rollback was needed. A run that makes the router more willing to
    # claim unrelated queries is undone in the same transaction.
    if report.decoys_after < report.decoys_before:
        for category, phrasings in teach.items():
            for phrase in phrasings:
                tokens = content_tokens(phrase)
                if tokens:
                    router.unlearn(" ".join(tokens), category)
        report.taught = {}
        report.trained = {}
        report.rolled_back = True
        report.decoys_after = _decoys_left_alone(router, categories)
    return report


def _catalogue_state(home: Path) -> dict:
    """The cross-run training ledger, stored beside the vault.

    One model per run only works if the runs share memory: which teachers have
    already taught, and how much of each category's budget is spent. Without
    it, every invocation would re-teach the same phrasings and the §11.13
    ceiling — which is about total absorbed weight, not per-run volume — would
    be exceeded by simple repetition, which is exactly how the second training
    run in §11.16 degraded the library.
    """
    import json

    path = home / "vault" / "distilled.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"teachers": [], "phrasings": {}}


def _save_catalogue_state(home: Path, state: dict) -> None:
    import json

    path = home / "vault" / "distilled.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True),
                    encoding="utf-8")


def train_next_voice(session, home: Path, holdout: float = 0.5,
                     seed: int = 0, ttl: int = 900) -> TrainingReport:
    """Train from the next untried voice, smallest model first.

    One model at a time, smallest to largest, until the catalogue is full —
    the shape the training was asked to take, and each part has a reason:

    * **One at a time** — the teachers here are 7-48B models on one card;
      loading several thrashes VRAM (measured earlier at ~78 GB resident with
      duplicate instances). Each run loads exactly one model.
    * **Smallest first** — cheapest generation first, so the library banks most
      of its coverage before the expensive models are ever loaded. The 7B costs
      seconds per category where the 35B cost 65 s.
    * **Until full** — "enough catalogued info" is made precise by the §11.13
      ceiling: each category absorbs at most
      :data:`SAFE_PHRASINGS_PER_CATEGORY` distilled phrasings **ever, across
      all runs**, tracked in ``vault/distilled.json``. When every trainable
      category is at budget, the run reports the catalogue full and teaches
      nothing.

    A voice that already taught is skipped — a lineage sibling of a used
    teacher adds near-identical phrasing distributions (§11.12), so the queue
    is voices, not models.

    Args:
        session: A built session, whose router is trained in place.
        home: The session directory, for the cross-run ledger.
        holdout: Fraction of new phrasings withheld and measured.
        seed: Split seed.
        ttl: Idle seconds before the teacher unloads.

    Returns:
        The :class:`TrainingReport`. ``model`` names the voice used, or is
        empty when the catalogue is already full or no voice remains.
    """
    from ultraquant.interpreter.llmls import catalogue, independent_groups

    report = TrainingReport()
    topics = _trainable_categories(session.router, report)
    if not topics:
        return report

    state = _catalogue_state(home)
    used = set(state.get("teachers", []))
    taught: dict = state.get("phrasings", {})

    remaining = {name: SAFE_PHRASINGS_PER_CATEGORY - len(taught.get(name, []))
                 for name in topics}
    open_categories = {name: room for name, room in remaining.items()
                       if room > 0}
    if not open_categories:
        report.skipped["(all)"] = (
            f"catalogue full: every category holds its measured budget of "
            f"{SAFE_PHRASINGS_PER_CATEGORY} distilled phrasings; further "
            "training adds nothing the decoy control would survive")
        return report

    cards = [c for c in catalogue() if c.is_chat]
    voices = independent_groups(cards)
    untried = []
    for group in voices:
        members = sorted(group, key=lambda c: (parameter_count(c.id), c.id))
        if any(c.id in used for c in group):
            continue
        untried.append(members[0])
    if not untried:
        report.skipped["(all)"] = (
            "every independent voice has already taught; the queue is voices, "
            "not models, because a lineage sibling adds near-identical "
            "phrasings")
        return report
    untried.sort(key=lambda c: (parameter_count(c.id), c.id))
    card = untried[0]
    report.model = card.id

    panel = TeacherPanel([card.id], client=LMStudioClient(timeout=600.0),
                         ttl=ttl)
    panel.load(card.id)

    already = {phrase.lower() for phrases in taught.values()
               for phrase in phrases}
    generated: dict[str, list[str]] = {}
    for category, room in sorted(open_categories.items()):
        topic = topics[category]
        produced = generate_phrasings(panel, category, topic,
                                      count=min(room + 2,
                                                SAFE_PHRASINGS_PER_CATEGORY))
        usable = [phrase for phrase in produced.phrasings
                  if content_tokens(phrase)
                  and phrase.lower() not in already][:room]
        if len(usable) >= 2:
            generated[category] = usable
        else:
            report.skipped[category] = (
                f"only {len(usable)} new usable phrasing(s) from this voice")

    report = _teach_and_measure(session, generated, report, holdout, seed)

    # The ledger records only what was actually taught, so a failed teach does
    # not consume budget, but the teacher is recorded either way - a voice that
    # produced nothing usable will produce nothing usable next run too.
    state.setdefault("teachers", []).append(card.id)
    # Only the taught half consumes budget. The first version recorded every
    # *generated* phrasing, so one teacher filled the whole ledger while half
    # of each category's real absorption budget sat unspent - and run 2 would
    # have reported "catalogue full" after a single voice.
    for category, phrasings in report.taught.items():
        kept = state.setdefault("phrasings", {}).setdefault(category, [])
        kept.extend(phrasings)
    _save_catalogue_state(home, state)
    return report


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
    requested = SAFE_PHRASINGS_PER_CATEGORY if count is None else int(count)
    per_category = min(requested, SAFE_PHRASINGS_PER_CATEGORY)

    report = TrainingReport(model=", ".join(card.id for card in panel.cards))
    topics = _trainable_categories(session.router, report)
    generated: dict[str, list[str]] = {}
    for category, topic in topics.items():
        try:
            produced = generate_phrasings(panel, category, topic,
                                          count=per_category)
        except LMStudioUnavailable as exc:
            report.skipped[category] = f"teacher unavailable: {exc}"
            continue
        usable = [p for p in produced.phrasings if content_tokens(p)]
        if len(usable) < 2:
            # generate_phrasings catches a per-teacher failure internally and
            # records it, so an empty result can mean the teacher never
            # answered. Reporting that as "produced nothing usable" points at
            # the prompt when the fix is the timeout - which is exactly what
            # happened: 65 s per category against a 120 s client default, and
            # every category came back "0 usable".
            why = next((reason for _who, reason in produced.rejected
                        if "unavailable" in reason or "timed out" in reason),
                       None)
            report.skipped[category] = (
                f"teacher did not answer: {why}" if why
                else f"only {len(usable)} usable phrasing(s) came back")
            continue
        generated[category] = usable
    return _teach_and_measure(session, generated, report, holdout, seed)


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
    parser.add_argument("--next", action="store_true", dest="next_voice",
                        help="train from the next untried voice, smallest "
                             "model first; cumulative budget tracked in "
                             "vault/distilled.json, so repeated runs stop at "
                             "the measured ceiling")
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
        if not args.next_voice and not models:
            from ultraquant.interpreter.llmls import catalogue

            models = [card.id for card in catalogue()
                      if card.is_chat and card.loaded]
            if not models:
                print("No chat model is loaded in LM Studio. Load one, or "
                      "pass --model.", file=sys.stderr)
                return 1
        # Generous: a 35B model took 65 s to list twelve phrasings for one
        # category on this machine, and the client's 120 s default left almost
        # no headroom once several categories ran in sequence.
        panel = (None if args.next_voice else
                 TeacherPanel(models, client=LMStudioClient(timeout=600.0)))
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
        if args.next_voice:
            report = train_next_voice(session, home)
        else:
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
