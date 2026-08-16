"""Does a reference index beat simply forgetting? The pre-registered gate.

A bounded context window has to do *something* when it overflows. The cheap
answer is to forget — which is what the existing working memory does, and what a
transformer's KV cache does when it fills. The claim under test is that keeping
a 24-byte reference per turn buys back most of what forgetting loses, for a cost
that stays flat as history grows.

**The task.** A fact is stated in an early turn, buried under enough later turns
that it has been evicted from the window, and then asked about. A window that
forgets cannot answer. A window that kept a reference should be able to fetch
it.

**The gate, written before the run.**

1. On **buried** questions, recall must beat the forgetting baseline by a margin
   larger than the seed-to-seed standard deviation — the §11.5 criterion.
2. The resident footprint must stay **within the declared byte budget**. A
   mechanism that answers more by quietly holding more has not solved the
   problem it was given; it has spent the constraint.

**The control, chosen so it can fail.** Recall is only interesting if it fetches
the *right* turn. So the same measurement runs on **distractor-heavy** history,
where many turns share vocabulary with the question but only one carries the
answer — and it is scored at ``top_k=1``, where a distractor out-ranking the
answer costs the point outright.

The strictness was not the first choice, and the first choice was wrong. Scored
at ``top_k=3`` the control sat at a ceilinged **1.000**: a query only had to
rank the answer in its top three, which nothing plausible could fail. That is
precisely the defect §11.11 was caught making, reappearing in the gate written
to avoid it.

A second, cheaper control: on questions whose answer is still *resident*, recall
must not make things worse. Fetching should add, never displace.

Run it::

    python -m ultraquant.experiments.context_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field

from ultraquant.memory.context import ContextWindow

__all__ = ["ContextGateReport", "run_gate"]

#: The pool facts are drawn from. Each pairs a distinctive subject with a value
#: and a question, so a correct recall is unambiguous to check.
#:
#: A pool, not a fixed list, because the first version shuffled a fixed six and
#: **every seed produced the identical delta**. Recall depends on vocabulary
#: overlap, which does not care what order turns arrived in, so the seeds were
#: varying nothing and the gate had no variance to measure its margin against.
#: It failed on exactly that, which is the guard working. Sampling a different
#: subset per seed makes each history genuinely different.
_FACT_POOL = [
    ("the tower height", "324 metres", "how tall is the tower"),
    ("the bridge length", "1280 metres", "how long is the bridge"),
    ("the vault capacity", "9400 shards", "what is the vault capacity"),
    ("the router latency", "4.2 milliseconds", "what is the router latency"),
    ("the archive checksum", "blake2b", "which checksum does the archive use"),
    ("the shard budget", "37 megabytes", "what is the shard budget"),
    ("the kernel width", "18 qubits", "how wide is the kernel"),
    ("the harbour depth", "12 fathoms", "how deep is the harbour"),
    ("the lattice spacing", "5 angstroms", "what is the lattice spacing"),
    ("the orbit period", "687 days", "how long is the orbit period"),
    ("the furnace temperature", "1450 celsius", "how hot is the furnace"),
    ("the ledger balance", "8200 credits", "what is the ledger balance"),
]

#: How many facts each history plants.
_FACTS_PER_RUN = 6

#: Filler that carries no answer. Two flavours: unrelated chatter, and
#: distractors that deliberately reuse the questions' vocabulary.
_CHATTER = [
    "the weather today is mild and clear",
    "I made coffee before starting work",
    "the meeting was moved to the afternoon",
    "there is a package waiting downstairs",
    "the printer needs more paper again",
]
_DISTRACTORS = [
    "the tower was repainted last spring",
    "the bridge was closed for maintenance",
    "the vault was inspected on tuesday",
    "the router was replaced last year",
    "the archive was moved to new storage",
    "the shard naming convention changed",
]


@dataclass
class ContextGateReport:
    """What the gate concluded.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        forgetting: Baseline accuracy — a window that simply drops overflow.
        recall: Accuracy with the reference index.
        delta: ``recall - forgetting``.
        seed_sd: Seed-to-seed standard deviation of the delta.
        margin_ratio: ``delta / seed_sd``.
        distractor_recall: Accuracy on the control condition, scored strictly at
            ``top_k=1``.
        resident_recall: Accuracy when the answer is still in the window.
        budget_held: Whether the resident footprint stayed inside its budget.
        peak_resident: Largest resident content seen, in bytes.
        index_bytes: Resident index size at the end of a run.
        stored_bytes: Bytes on disk at the end of a run.
        reason: Plain-language verdict.
    """

    passes: bool = False
    forgetting: float = 0.0
    recall: float = 0.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    distractor_recall: float = 0.0
    resident_recall: float = 0.0
    budget_held: bool = True
    peak_resident: int = 0
    index_bytes: int = 0
    stored_bytes: int = 0
    turns: int = 0
    reason: str = ""

    def as_text(self) -> str:
        """A printable report, including what argues against passing."""
        lines = [
            f"buried questions      forgetting {self.forgetting:.3f}   "
            f"recall {self.recall:.3f}   delta {self.delta:+.3f}",
            f"distractors (CONTROL) recall {self.distractor_recall:.3f}"
            "   <- top_k=1 over vocabulary-sharing filler; can fall to 0",
            f"resident (CONTROL)    recall {self.resident_recall:.3f}"
            "   <- fetching must not displace what is already held",
            f"seed sd {self.seed_sd:.3f}   margin ratio "
            f"{self.margin_ratio:.2f}x  (gate needs > 1.0)",
            f"footprint  peak resident {self.peak_resident:,} B   "
            f"index {self.index_bytes:,} B   on disk {self.stored_bytes:,} B "
            f"over {self.turns} turns",
        ]
        if not self.budget_held:
            lines.append("BUDGET EXCEEDED - answering more by holding more is "
                         "not a solution")
        lines.append(("PASS - " if self.passes else "FAIL - ") + self.reason)
        return "\n".join(lines)


def _conversation(facts: list, rng: random.Random, filler: list[str],
                  depth: int) -> list[str]:
    """One history: the sampled facts stated, then buried under filler."""
    turns = [f"{subject} is {value}" for subject, value, _q in facts]
    rng.shuffle(turns)
    for _ in range(depth):
        turns.append(rng.choice(filler))
    return turns


def _score(window: ContextWindow, facts: list, use_recall: bool,
           top_k: int = 3) -> float:
    """Fraction of facts answerable from what the window can reach.

    ``top_k`` matters for the control: at 3, a query only has to rank the
    answer in its top three, which the distractor condition passed at a
    ceilinged 1.000 - a control that cannot fail measures nothing. The strict
    control uses ``top_k=1``, where a distractor out-ranking the answer costs
    the point outright.
    """
    hit = 0
    for subject, value, question in facts:
        reachable = list(window.resident())
        if use_recall:
            reachable += window.recall(question, top_k=top_k)
        if any(value in str(turn.get("text", "")) for turn in reachable):
            hit += 1
    return hit / len(facts)


def _run_once(filler: list[str], depth: int, budget: int,
              seed: int) -> tuple[float, float, dict]:
    """Build one history and score both arms over it.

    Both arms read the *same* window, so the only difference is whether recall
    is consulted. Building two windows would let disk layout differ between
    them and confound the comparison.
    """
    rng = random.Random(seed)
    facts = rng.sample(_FACT_POOL, _FACTS_PER_RUN)
    root = tempfile.mkdtemp(prefix="uq_ctx_")
    try:
        window = ContextWindow(root, budget_bytes=budget)
        peak = 0
        for text in _conversation(facts, rng, filler, depth):
            window.add(text)
            peak = max(peak, window.stats()["resident_bytes"])
        forgetting = _score(window, facts, use_recall=False)
        recall = _score(window, facts, use_recall=True)
        stats = window.stats()
        stats["peak_resident"] = peak
        stats["strict"] = _score(window, facts, use_recall=True, top_k=1)
        return forgetting, recall, stats
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_gate(seeds: int = 8, depth: int = 60,
             budget: int = 512) -> ContextGateReport:
    """Run the gate.

    Args:
        seeds: Histories to average over. Must exceed 1 or the sd is meaningless.
        depth: Filler turns piled on after the facts. Large enough that the
            facts are certainly evicted at this budget.
        budget: Resident content budget in bytes. Small on purpose — the whole
            question is what happens when the window is the scarce thing.

    Returns:
        The :class:`ContextGateReport`.
    """
    forgetting_scores, recall_scores, deltas = [], [], []
    distractor_scores, resident_scores = [], []
    peak = 0
    stats: dict = {}
    for seed in range(seeds):
        forgetting, recall, stats = _run_once(_CHATTER, depth, budget, seed)
        forgetting_scores.append(forgetting)
        recall_scores.append(recall)
        deltas.append(recall - forgetting)
        peak = max(peak, stats["peak_resident"])

        # Control 1: filler that shares the questions' vocabulary, scored at
        # top_k=1 so a distractor out-ranking the answer actually costs a point.
        _f, _r, control_stats = _run_once(_DISTRACTORS, depth, budget, seed)
        distractor_scores.append(control_stats["strict"])

        # Control 2: nothing buried, so the answers are still resident.
        _f2, resident, _s2 = _run_once(_CHATTER, 0, budget, seed)
        resident_scores.append(resident)

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: these seeds are a sample of possible histories, and
    # dividing by n rather than n-1 would understate the spread and inflate the
    # margin against the gate's own threshold.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    report = ContextGateReport(
        forgetting=statistics.fmean(forgetting_scores),
        recall=statistics.fmean(recall_scores),
        delta=delta,
        seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        distractor_recall=statistics.fmean(distractor_scores),
        resident_recall=statistics.fmean(resident_scores),
        budget_held=peak <= budget,
        peak_resident=peak,
        index_bytes=stats.get("index_bytes", 0),
        stored_bytes=stats.get("stored_bytes", 0),
        turns=stats.get("turns", 0),
    )

    # Budget first: a mechanism that spent its constraint has not met it.
    if not report.budget_held:
        report.reason = (f"resident content peaked at {peak:,} B against a "
                         f"{budget:,} B budget")
        return report
    if delta <= 0:
        report.reason = f"recall did not beat forgetting ({delta:+.3f})"
        return report
    if seed_sd <= 0.0:
        report.reason = (f"every history gave the same delta ({delta:+.3f}), so "
                         "there is no variance to measure the margin against")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"margin {delta:+.3f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    if report.distractor_recall < 0.5:
        report.reason = (
            f"the control failed: with vocabulary-sharing distractors recall "
            f"fell to {report.distractor_recall:.3f}, so the sketch is fetching "
            "something plausible rather than the right turn")
        return report
    if report.resident_recall < 1.0:
        report.reason = (
            f"the resident control failed at {report.resident_recall:.3f}: "
            "consulting recall must not displace what the window already holds")
        return report
    report.passes = True
    report.reason = (
        f"buried recall {report.forgetting:.3f} -> {report.recall:.3f} "
        f"({delta:+.3f}, {report.margin_ratio:.2f}x seed sd), controls held "
        f"({report.distractor_recall:.3f} distractor, "
        f"{report.resident_recall:.3f} resident), and the window stayed inside "
        f"its {budget:,} B budget")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
