"""Reading until you know enough. The adaptive-recall gate.

A transformer spends its entire parameter space on every token.
Asked "what is 2 + 2" it does the same billions of multiply-
accumulates it does for a question at the edge of its competence,
because a dense forward pass has no way to do less. **Cost is a
property of the model, not of the question.**

The claim this gate tests is the project's own thesis: that a
catalogued library does not have to work that way - it can read the
shards the question points at, and stop when it knows enough. The
library has had the routing and the paging for a long time.
§11.104 adds the stopping, and stopping is the half with no analogue
in a dense model.

**Two standard options, and both are real.**

* **Read everything** - consult every candidate the router offers.
  This is what a dense model does, and it is the accuracy ceiling:
  nothing that reads less can beat it, so the question is how much
  less can be read before accuracy moves.
* **Read a fixed top-k** - what retrieval augmentation does. This is
  the harder competitor, because it also reads less. Adaptive recall
  has to beat a fixed k *matched to its own mean spend*, or it is an
  elaborate way of doing what a constant would have done.

**The criteria, written before the run.**

1. **Cost scales with the question** - the number of reads must
   VARY across inputs, by more than one read of spread. A system
   whose spread is zero has a fixed cost and has not demonstrated
   the claim, however low that cost is. This is the criterion that
   distinguishes the thesis from a smaller constant.
2. **Accuracy holds against reading everything** - adaptive accuracy
   must be within one standard error of the full-scan ceiling. Cheap
   and wrong is not the claim.
3. **It beats a fixed top-k matched to its own mean spend** -
   strictly higher accuracy at the same average number of reads, by
   more than the seed-to-seed sd. If a constant does as well, the
   loop is decoration.
4. **The savings are real and reported** - mean reads, mean paid
   reads (cache misses only), and the share of offered experts never
   touched. A saving measured in consultations that were all cache
   hits is not a saving.
5. **Stopping early is not giving up** - among recalls that stopped
   on "plateau", the full scan must not have found the answer
   either, in at least 80% of cases. A plateau that abandons
   answerable questions is a bug wearing a policy's clothes.

**FAILED. The sharding half of the thesis is vindicated; the
entropy-stopping half is refuted, and the two results are worth
separating carefully because they point opposite ways.**

**What passed, and it is the bigger claim.** Reading a fixed **3 of
12** experts scores **0.9900** - identical to reading all twelve.
Four gets 0.9917 and twelve gets 0.9900; the curve is flat from k=3
onward.

| experts read | 1 | 2 | **3** | 4 | 12 |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.9450 | 0.9733 | **0.9900** | 0.9917 | 0.9900 |

**A quarter of the library buys all of the accuracy.** That is the
claim a dense model cannot make: a transformer spends 100% of its
parameters on every token because it has no catalog to consult. The
measurement says the catalog is worth what it was always claimed to
be worth.

**What failed is the loop.** Adaptive recall was beaten by a plain
constant, and it was beaten three different ways:

| comparison | adaptive | fixed k | verdict |
|---|---:|---:|:--|
| round(mean reads) = 3 | 0.9667 | 0.9833 | lost |
| sweep, 12 settings of router noise and floor | - | - | lost at **11 of 12** |
| curve interpolated at adaptive's own 2.54 reads | 0.9667 | **0.9823** | lost |

The third row is the one that matters, because run two's comparison
was unfair in adaptive's FAVOUR - `round(2.54) = 3` gave the
constant a bigger budget than the thing it was competing with.
Matched properly, adaptive still loses by 0.0156.

**Run one failed differently, and the fix did not rescue it.** The
loop first returned the first expert to clear the floor, and lost at
eleven of twelve swept settings. The sweep showed why: first-past-
the-post against best-of-k is not a contest about cost, it is a
contest about **comparison**, and stopping at the first adequate
answer throws away the very thing the constant was quietly buying.
Run two made the loop keep the best and confirm nothing better
follows. It read more (1.25 to 2.54) and still lost.

**Criterion 1 also failed, and says the same thing from the other
side**: reads varied with sd 0.64 then 0.76, under the one-read bar.
The cost is very nearly constant - so the loop is an expensive way
of computing a number that could have been a parameter.

**Criterion 5 failed outright.** Of recalls that stopped on
"plateau", only **11.8%** were questions the full scan could not
answer either. The plateau rule abandons answerable questions almost
nine times in ten. It is a bug wearing a policy's clothes, which is
what the criterion was written to detect.

**The limitation, stated as a limitation and not as a defence.** In
this world every question has the same shape - which of twelve
categories, then which of four labels - so questions do not differ
in how much reading they need. Adaptive stopping can only pay where
that varies. Building a world with heterogeneous difficulty and
reporting THAT would be searching for a setting that passes, which
is the move the sweep exists to refuse. The honest statement is:
**on questions of uniform difficulty, a constant k is better than
entropy-driven stopping, and the condition under which stopping
might pay is untested here.**

Run it::

    python -m ultraquant.experiments.enough_gate
"""

from __future__ import annotations

import math
import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.enough import recall_until_enough
from ultraquant.shards.vault import ShardVault

__all__ = ["EnoughReport", "run_gate"]


#: Feature width shared by every expert, so routing errors are
#: possible rather than caught by a shape check.
_WIDTH = 12

#: Labels each category's expert distinguishes.
_LABELS = ["alpha", "beta", "gamma", "delta"]


def _world(categories: int, seed: int):
    """A library where each expert knows its own corner and no other.

    Every category owns a direction in the shared feature space; a
    sample is that direction, plus a label-specific offset, plus
    noise. An expert trained on one category therefore has nothing
    useful to say about another's inputs - which is exactly what
    makes evidence a usable stopping signal rather than a formality.
    """
    rng = random.Random(seed)
    signatures = []
    for _ in range(categories):
        vector = [rng.gauss(0, 1) for _ in range(_WIDTH)]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        signatures.append([v / norm for v in vector])
    offsets = [[rng.gauss(0, 1) for _ in range(_WIDTH)]
               for _ in _LABELS]
    return signatures, offsets


def _sample(signatures, offsets, category: int, label: int,
            rng: random.Random, noise: float = 0.35) -> list:
    return [3.0 * signatures[category][i] + offsets[label][i]
            + rng.gauss(0, noise) for i in range(_WIDTH)]


def _ranking(signatures, features: list, rng: random.Random,
             confusion: float) -> list:
    """The router's guess at which categories are worth reading.

    Deliberately imperfect: the score is cosine to each signature
    plus noise, so the true category is usually near the front and
    sometimes is not. A ranking that were always right would make
    adaptive recall trivially optimal and would measure nothing.
    """
    norm = math.sqrt(sum(v * v for v in features)) or 1.0
    scored = []
    for index, signature in enumerate(signatures):
        dot = sum(a * b for a, b in zip(signature, features)) / norm
        scored.append((dot + rng.gauss(0, confusion), index))
    scored.sort(reverse=True)
    return [index for _score, index in scored]


@dataclass
class EnoughReport:
    """What stopping early cost, and what it saved.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        seeds: Independent worlds measured.
        categories: Experts each question could have consulted.
        questions: Test inputs per seed.
        adaptive_accuracy: Accuracy of the stopping loop.
        full_accuracy: Accuracy of reading every candidate.
        fixed_accuracy: Accuracy of a fixed top-k at the same spend.
        fixed_k: The k that matched adaptive's mean spend.
        mean_reads: Mean experts consulted by the loop.
        read_spread: Standard deviation of that, across questions.
        mean_paid: Mean consultations that were cache misses.
        spared: Share of offered experts never touched.
        stops: Reason -> share of recalls that ended that way.
        plateau_justified: Share of plateaus the full scan also missed.
        accuracy_sd: Seed-to-seed sd of the adaptive-minus-fixed gap.
        reason: Plain-language verdict.
    """

    passes: bool
    seeds: int = 0
    categories: int = 0
    questions: int = 0
    adaptive_accuracy: float = 0.0
    full_accuracy: float = 0.0
    fixed_accuracy: float = 0.0
    fixed_k: int = 0
    mean_reads: float = 0.0
    read_spread: float = 0.0
    mean_paid: float = 0.0
    spared: float = 0.0
    stops: dict = field(default_factory=dict)
    plateau_justified: float = 0.0
    accuracy_sd: float = 0.0
    reason: str = ""


def _build(root: Path, signatures, offsets, seed: int, budget: int):
    """Train one expert per category and leave them on storage."""
    vault = ShardVault(root / f"v{seed}")
    cache = ShardCache(budget)
    pool = ExpertPool(vault, cache, input_dim=_WIDTH, hidden=(16,),
                      seed=seed)
    rng = random.Random(9000 + seed)
    for category in range(len(signatures)):
        name = f"cat{category}"
        pool.ensure_expert(name, list(_LABELS))
        xs, ys = [], []
        for label in range(len(_LABELS)):
            for _ in range(40):
                xs.append(_sample(signatures, offsets, category, label, rng))
                ys.append(label)
        pool.train_expert(name, xs, ys, list(_LABELS), epochs=30, lr=0.05)
    return pool


def _best_of(pool, order: list, features: list, limit: int,
             floor: float) -> tuple:
    """Consult `limit` experts and keep the most confident answer.

    This is both standard options: `limit = len(order)` is reading
    everything, and a smaller limit is fixed top-k. Neither stops
    early, so both pay their full price on every question.
    """
    best_label, best_evidence = None, -1.0
    for category in order[:limit]:
        try:
            label, result = pool.predict_uncertain(f"cat{category}",
                                                   features, floor)
        except (KeyError, ValueError):
            continue
        if result.evidence > best_evidence:
            best_label, best_evidence = label, result.evidence
    return best_label, best_evidence


def run_gate(seeds: int = 5, categories: int = 12, questions: int = 120,
             floor: float = 1.2, confusion: float = 0.35,
             budget: int = 1 << 22) -> EnoughReport:
    """Ask each question three ways and compare cost against answer."""
    root = Path(tempfile.mkdtemp(prefix="uq_enough_"))
    adaptive_acc, full_acc, fixed_acc = [], [], []
    gaps, reads_all, paid_all, spared_all = [], [], [], []
    stops: dict = {}
    plateau_hits = plateau_total = 0
    chosen_k = 0
    try:
        for seed in range(seeds):
            signatures, offsets = _world(categories, seed=seed)
            pool = _build(root, signatures, offsets, seed, budget)
            rng = random.Random(4000 + seed)

            asked = []
            for _ in range(questions):
                category = rng.randrange(categories)
                label = rng.randrange(len(_LABELS))
                features = _sample(signatures, offsets, category, label, rng)
                asked.append((features,
                              _ranking(signatures, features, rng, confusion),
                              _LABELS[label]))

            # Adaptive, and the reads it spent.
            adaptive_right = 0
            reads: list = []
            plateau_cases: list = []
            for features, order, truth in asked:
                recall = recall_until_enough(
                    pool, [f"cat{c}" for c in order], features, floor)
                reads.append(recall.reads)
                paid_all.append(recall.paid_reads)
                spared_all.append(recall.spared / max(1, recall.available))
                stops[recall.stopped] = stops.get(recall.stopped, 0) + 1
                if recall.label == truth:
                    adaptive_right += 1
                if recall.stopped == "plateau":
                    plateau_cases.append((features, order, truth))
            adaptive_acc.append(adaptive_right / len(asked))
            reads_all.extend(reads)
            mean_reads = statistics.fmean(reads)

            # Reading everything: the ceiling.
            full_right = sum(
                1 for features, order, truth in asked
                if _best_of(pool, order, features, len(order),
                            floor)[0] == truth)
            full_acc.append(full_right / len(asked))

            # Fixed top-k matched to adaptive's own mean spend.
            k = max(1, min(categories, round(mean_reads)))
            chosen_k = k
            fixed_right = sum(
                1 for features, order, truth in asked
                if _best_of(pool, order, features, k, floor)[0] == truth)
            fixed_acc.append(fixed_right / len(asked))
            gaps.append(adaptive_acc[-1] - fixed_acc[-1])

            # Criterion 5: was each plateau a question the full scan
            # could not answer either?
            for features, order, truth in plateau_cases:
                plateau_total += 1
                if _best_of(pool, order, features, len(order),
                            floor)[0] != truth:
                    plateau_hits += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return _finish(seeds, categories, questions, adaptive_acc, full_acc,
                   fixed_acc, gaps, reads_all, paid_all, spared_all,
                   stops, plateau_hits, plateau_total, chosen_k)


def _finish(seeds, categories, questions, adaptive_acc, full_acc,
            fixed_acc, gaps, reads_all, paid_all, spared_all, stops,
            plateau_hits, plateau_total, chosen_k) -> EnoughReport:
    """Apply the criteria as they were written."""
    adaptive = statistics.fmean(adaptive_acc)
    full = statistics.fmean(full_acc)
    fixed = statistics.fmean(fixed_acc)
    gap = statistics.fmean(gaps)
    gap_sd = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
    mean_reads = statistics.fmean(reads_all)
    spread = statistics.pstdev(reads_all) if len(reads_all) > 1 else 0.0
    full_sd = (statistics.pstdev(full_acc) if len(full_acc) > 1 else 0.0)
    standard_error = full_sd / math.sqrt(len(full_acc)) if full_acc else 0.0

    varies = spread > 1.0
    holds = adaptive >= full - max(standard_error, 1e-9)
    beats_fixed = gap > gap_sd
    total_stops = sum(stops.values()) or 1
    justified = (plateau_hits / plateau_total) if plateau_total else 1.0
    plateau_ok = justified >= 0.80
    passes = varies and holds and beats_fixed and plateau_ok

    report = EnoughReport(
        passes=passes, seeds=seeds, categories=categories,
        questions=questions, adaptive_accuracy=adaptive,
        full_accuracy=full, fixed_accuracy=fixed, fixed_k=chosen_k,
        mean_reads=mean_reads, read_spread=spread,
        mean_paid=statistics.fmean(paid_all) if paid_all else 0.0,
        spared=statistics.fmean(spared_all) if spared_all else 0.0,
        stops={k: v / total_stops for k, v in stops.items()},
        plateau_justified=justified, accuracy_sd=gap_sd)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {seeds} worlds of "
        f"{categories} experts, {questions} questions each; adaptive reads "
        f"{mean_reads:.2f} of {categories} (sd {spread:.2f}, "
        f"{'varies' if varies else 'FIXED COST'}), {report.mean_paid:.2f} "
        f"paid, sparing {report.spared:.1%}; accuracy {adaptive:.4f} "
        f"adaptive against {full:.4f} reading everything "
        f"({'holds' if holds else 'LOST ACCURACY'}) and {fixed:.4f} for "
        f"fixed top-{chosen_k} (gap {gap:+.4f} at seed sd {gap_sd:.4f}, "
        + ("beats the constant" if beats_fixed
           else "NOT better than a constant")
        + f"); plateaus justified {justified:.1%} "
        f"({'honest' if plateau_ok else 'GIVING UP EARLY'})")
    return report


def cost_curve(seeds: int = 5, categories: int = 12, questions: int = 120,
               floor: float = 1.2, confusion: float = 0.35,
               budget: int = 1 << 22) -> dict:
    """Adaptive's (reads, accuracy) point against the whole fixed-k curve.

    Criterion 3 asks for a fixed k "matched to its own mean spend",
    and run two implemented that as `round(mean_reads)`. At a mean of
    2.54 that rounds to 3, which reads MORE than adaptive does - so
    the constant was being given a larger budget than the thing it
    was competing with. Matching properly means interpolating the
    fixed-k curve at adaptive's actual mean, which is what this does.

    Returns the curve, adaptive's point, and the interpolated rival.
    """
    root = Path(tempfile.mkdtemp(prefix="uq_curve_"))
    per_k: dict = {k: [] for k in range(1, categories + 1)}
    adaptive_points: list = []
    try:
        for seed in range(seeds):
            signatures, offsets = _world(categories, seed=seed)
            pool = _build(root, signatures, offsets, seed, budget)
            rng = random.Random(4000 + seed)
            asked = []
            for _ in range(questions):
                category = rng.randrange(categories)
                label = rng.randrange(len(_LABELS))
                features = _sample(signatures, offsets, category, label, rng)
                asked.append((features,
                              _ranking(signatures, features, rng, confusion),
                              _LABELS[label]))
            right, reads = 0, []
            for features, order, truth in asked:
                recall = recall_until_enough(
                    pool, [f"cat{c}" for c in order], features, floor)
                reads.append(recall.reads)
                right += int(recall.label == truth)
            adaptive_points.append((statistics.fmean(reads),
                                    right / len(asked)))
            for k in per_k:
                hits = sum(1 for features, order, truth in asked
                           if _best_of(pool, order, features, k,
                                       floor)[0] == truth)
                per_k[k].append(hits / len(asked))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    curve = {k: statistics.fmean(v) for k, v in per_k.items()}
    mean_reads = statistics.fmean(p[0] for p in adaptive_points)
    adaptive = statistics.fmean(p[1] for p in adaptive_points)
    low = max(1, int(mean_reads))
    high = min(categories, low + 1)
    weight = mean_reads - low
    rival = curve[low] * (1 - weight) + curve[high] * weight
    return {"curve": curve, "reads": mean_reads, "adaptive": adaptive,
            "rival": rival, "gap": adaptive - rival}


def where_it_pays(confusions: tuple = (0.35, 0.8, 1.5, 3.0),
                  floors: tuple = (1.2, 1.6, 1.9),
                  seeds: int = 3, categories: int = 12,
                  questions: int = 80) -> list:
    """Sweep the conditions rather than search for one that passes.

    Run one FAILED at a single setting where the router was nearly
    perfect - fixed top-1 scored 0.9450, so there was almost nothing
    for stopping to win. The tempting response is to find a harder
    world and report that. The honest one is to vary the difficulty
    across its whole range and report the CURVE, because "adaptive
    recall beats a constant" is not true or false in general: it is
    true where routing is unreliable enough that a constant k is
    either wasteful or wrong, and the useful result is where that
    boundary sits.

    Returns one row per (confusion, floor) with both accuracies and
    the mean reads, so the trade is visible at every point.
    """
    rows: list = []
    for confusion in confusions:
        for floor in floors:
            report = run_gate(seeds=seeds, categories=categories,
                              questions=questions, floor=floor,
                              confusion=confusion)
            rows.append({
                "confusion": confusion, "floor": floor,
                "reads": report.mean_reads, "spread": report.read_spread,
                "adaptive": report.adaptive_accuracy,
                "fixed": report.fixed_accuracy, "k": report.fixed_k,
                "full": report.full_accuracy,
                "gap": report.adaptive_accuracy - report.fixed_accuracy,
                "sd": report.accuracy_sd,
            })
    return rows


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"worlds                {result.seeds} x {result.categories} experts")
    print(f"questions per world   {result.questions}")
    print()
    print(f"adaptive reads        {result.mean_reads:.2f} of "
          f"{result.categories}  (sd {result.read_spread:.2f})")
    print(f"paid reads (misses)   {result.mean_paid:.2f}")
    print(f"experts never touched {result.spared:.1%}")
    print()
    print(f"accuracy, adaptive    {result.adaptive_accuracy:.4f}")
    print(f"accuracy, read all    {result.full_accuracy:.4f}")
    print(f"accuracy, fixed top-{result.fixed_k}  {result.fixed_accuracy:.4f}")
    print()
    for name, share in sorted(result.stops.items()):
        print(f"  stopped {name:<10} {share:.1%}")
    print(f"  plateaus justified {result.plateau_justified:.1%}")
    print()
    print(result.reason)
