"""A prediction layer that says how much it knows. The entropy gate.

`UltraQuantNet.predict` returns the argmax and its softmax
probability. That is the standard thing to report and the standard
thing to be misled by: an argmax always exists, so a network that
has learned nothing still names a class and still puts a number
beside it. Shown something that is not a glyph at all, the
recognizer answers "diamond, 0.31" and nothing in the reply says the
question was never answerable.

That breaks a line the rest of this system keeps everywhere else.
§11.48 refuses to answer "is X Y?" about a subject nobody holds -
absence of belief is not belief in absence - and the pipeline says
"I don't know" in a dozen places. The prediction layer was the one
component that could not, because its output shape had no room for
it.

`model/uncertainty.py` gives it room: the whole distribution's
entropy in BITS, the evidence ``log2(n) - H`` that grows with
knowledge rather than shrinking with it, the margin to the
runner-up, and a floor below which the layer declines.

**Measured against the standard option, which is not a strawman.**
Maximum softmax probability is what almost everything reports, it is
monotonic with entropy for two classes, and for a peaked
distribution the two agree closely. So the question is not whether
entropy is more sophisticated - it is whether it actually rejects
errors better, and the honest possibility is that it does not.

**The criteria, written before the run.**

1. **The measures are their definitions** - entropy is ``log2(n)``
   for a uniform distribution, 0 for a one-hot one, and matches a
   50-digit reference elsewhere. One value outside 1e-12 is a fail.
2. **Entropy rejects errors better than max-softmax** - ranking
   predictions by uncertainty and discarding the worst fifth, the
   error rate among what remains must be LOWER under entropy than
   under max-softmax, by more than the seed-to-seed sd across five
   seeds. This is the criterion that can genuinely fail, and if it
   does the honest answer is that the standard measure was already
   good enough here.
3. **A network that has learned nothing says so** - an untrained
   recognizer must average below 0.25 of its 3.00 bits. If random
   weights report high evidence, the measure is not measuring
   knowledge and nothing else it says means anything.
4. **Things that are not glyphs are declined** - at the measured
   floor, the decline rate on uniform-random inputs must be at
   least three times the decline rate on real noisy glyphs. A layer
   equally confident about noise and about a plus has learned
   nothing about itself.
5. **The floor is fitted on one split and reported on another** -
   coverage and accepted-error-rate come from data that did not
   choose the threshold. A floor tuned and reported on the same set
   is a confidence interval invented on the spot.

**FAILED criterion 2. Entropy is the WORST of the three measures at
the job it was proposed for**, and finding that out is the whole
value of having run the standard option beside it.

5 seeds, accuracy 0.9250. Ranking the same predictions and
discarding the least certain fifth:

| ranker | error at 80% coverage |
|---|---:|
| margin | **0.0078** |
| max-softmax | 0.0130 |
| entropy | 0.0260 |

Entropy loses to the plain max-softmax it was meant to improve on,
and loses again to `margin`, which came from the same new module.
The reason is structural rather than incidental: **entropy is moved
by the whole tail**, so a confident, correct prediction with a
little probability spread over the other seven classes scores as
uncertain, while max-softmax and margin look only at the top - which
is what actually decides whether the argmax is right.

**Criteria 3, 4 and 5 passed, and they are what entropy is
actually for.**

| | |
|---|---:|
| untrained evidence | **0.184 of 3.00 bits** |
| trained evidence | 2.248 of 3.00 bits |
| floor fitted on the fit split | 1.774 bits |
| coverage on the test split | 79.8% |
| error among accepted | **0.0178** |
| error had nothing been declined | 0.0750 |
| non-glyph noise declined | **98.8%** |
| real glyphs declined | 20.2% |

Declining the least-certain fifth cuts the error rate **4.2x**. And
the untrained number is the one that justifies the unit: a network
with random weights reports **0.184 of 3.00 bits** - it says it
knows nothing. Max-softmax on the same predictions reads 0.191,
which only means anything once the reader knows that chance is 0.125
on eight classes, and that comparison is not in the number.

**So the measures divide by which failure they catch**, measured at
matched coverage:

| floor on | coverage | accepted error | noise declined |
|---|---:|---:|---:|
| evidence | 81.5% | 0.0229 | **98.5%** |
| margin | 82.3% | **0.0077** | 93.8% |
| weaker of both | 82.5% | 0.0222 | 98.5% |

**`margin` catches wrong predictions; `evidence` catches inputs the
network has never seen.** A hard in-distribution case is a two-way
confusion - small margin, respectable evidence. An
out-of-distribution input spreads everywhere - low evidence, and a
margin that can still be accidentally wide. Both floors ship,
separately, because neither is the other.

**Requiring both was tried and does not compose.** Taking the weaker
of the two lands at evidence's error rate with none of margin's
gain, because whichever measure ranks a sample lower dominates and
that is usually evidence. Recorded so it is not retried.

That last table was run AFTER the criteria were fixed, so it is
reported rather than gated. The registered question was "does
entropy beat max-softmax", the answer is no, and dressing up the
follow-up as a pass would be scoring the coin after the throw.

Run it::

    python -m ultraquant.experiments.entropy_gate
"""

from __future__ import annotations

import math
import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.model import uncertainty
from ultraquant.pattern.recognition import LABELS, PatternRecognizer, make_dataset

__all__ = ["EntropyReport", "run_gate", "risk_at_coverage"]


def risk_at_coverage(scores: list, wrong: list, coverage: float) -> float:
    """Error rate among the most-certain `coverage` share.

    `scores` is an UNCERTAINTY - lower means more certain - so the
    kept set is the smallest scores. Ties are broken by the original
    order rather than by outcome, because sorting by anything the
    outcome touches would be scoring the coin after seeing it.
    """
    if not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    keep = max(1, int(round(len(order) * coverage)))
    kept = order[:keep]
    return sum(1 for i in kept if wrong[i]) / len(kept)


@dataclass
class EntropyReport:
    """What the whole distribution knew that its tallest bar did not.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        seeds: How many independent recognizers were measured.
        accuracy: Mean accuracy over the test split.
        risk_entropy: Mean error rate at 80% coverage, ranked by entropy.
        risk_confidence: The same, ranked by max-softmax.
        risk_gap: confidence minus entropy - positive means entropy won.
        risk_sd: Seed-to-seed sd of that gap.
        untrained_bits: Mean evidence of an untrained recognizer.
        trained_bits: Mean evidence of a trained one.
        floor: The evidence floor fitted on the fit split.
        coverage: Share of the test split accepted at that floor.
        accepted_error: Error rate among accepted test predictions.
        declined_all: Error rate had nothing been declined.
        noise_declined: Decline rate on uniform-random inputs.
        glyph_declined: Decline rate on real noisy glyphs.
        reason: Plain-language verdict.
    """

    passes: bool
    seeds: int = 0
    accuracy: float = 0.0
    risk_entropy: float = 0.0
    risk_confidence: float = 0.0
    risk_gap: float = 0.0
    risk_sd: float = 0.0
    untrained_bits: float = 0.0
    trained_bits: float = 0.0
    floor: float = 0.0
    coverage: float = 0.0
    accepted_error: float = 0.0
    declined_all: float = 0.0
    noise_declined: float = 0.0
    glyph_declined: float = 0.0
    reason: str = ""


def _measure(seed: int, root: Path, epochs: int):
    """One recognizer: train it, then read every prediction it makes."""
    recognizer = PatternRecognizer(mode="classical", seed=seed,
                                   memory_path=root / f"m{seed}")
    untrained_xs, _untrained_ys = make_dataset(4, 2, seed=500 + seed)
    untrained = [recognizer.net.predict_with_uncertainty(
        recognizer.features(x)).evidence for x in untrained_xs]

    recognizer.train(epochs=epochs, n_per_class=30, flips=2, seed=1 + seed)

    def read(xs, ys):
        rows = []
        for x, y in zip(xs, ys):
            result = recognizer.net.predict_with_uncertainty(
                recognizer.features(x))
            rows.append((result, result.label != y))
        return rows

    fit_xs, fit_ys = make_dataset(12, 3, seed=700 + seed)
    test_xs, test_ys = make_dataset(12, 3, seed=900 + seed)
    return recognizer, untrained, read(fit_xs, fit_ys), read(test_xs, test_ys)


def _noise_inputs(count: int, seed: int) -> list:
    """Inputs that are not glyphs: uniform random, no structure at all."""
    rng = random.Random(seed)
    return [[rng.random() for _ in range(25)] for _ in range(count)]


def run_gate(seeds: int = 5, epochs: int = 30,
             coverage: float = 0.80) -> EntropyReport:
    """Train several recognizers and read what their outputs knew."""
    root = Path(tempfile.mkdtemp(prefix="uq_entropy_"))
    try:
        gaps: list = []
        risks_e: list = []
        risks_c: list = []
        accuracies: list = []
        untrained_bits: list = []
        trained_bits: list = []
        floors: list = []
        coverages: list = []
        accepted: list = []
        all_error: list = []
        noise_rate: list = []
        glyph_rate: list = []

        for seed in range(seeds):
            recognizer, untrained, fit, test = _measure(seed, root, epochs)
            untrained_bits.append(statistics.fmean(untrained))
            trained_bits.append(statistics.fmean(r.evidence for r, _w in test))
            accuracies.append(1.0 - sum(w for _r, w in test) / len(test))

            # Criterion 2: the same predictions, ranked two ways.
            wrong = [w for _r, w in test]
            risk_e = risk_at_coverage([r.entropy for r, _w in test],
                                      wrong, coverage)
            risk_c = risk_at_coverage([-r.probability for r, _w in test],
                                      wrong, coverage)
            risks_e.append(risk_e)
            risks_c.append(risk_c)
            gaps.append(risk_c - risk_e)

            # Criterion 5: the floor is chosen on `fit` and every
            # number reported about it comes from `test`.
            fit_bits = sorted(r.evidence for r, _w in fit)
            index = min(len(fit_bits) - 1,
                        int(round(len(fit_bits) * (1.0 - coverage))))
            floor = fit_bits[index]
            floors.append(floor)
            kept = [(r, w) for r, w in test if r.evidence >= floor]
            coverages.append(len(kept) / len(test))
            accepted.append(sum(w for _r, w in kept) / len(kept)
                            if kept else 0.0)
            all_error.append(sum(wrong) / len(wrong))

            # Criterion 4: noise against real glyphs, same floor.
            noise = _noise_inputs(96, seed=1300 + seed)
            declined = sum(1 for x in noise
                           if recognizer.net.predict_with_uncertainty(
                               recognizer.features(x), floor).declined)
            noise_rate.append(declined / len(noise))
            glyph_rate.append(sum(1 for r, _w in test
                                  if r.evidence < floor) / len(test))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    gap = statistics.fmean(gaps)
    sd = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
    untrained = statistics.fmean(untrained_bits)
    noise_declined = statistics.fmean(noise_rate)
    glyph_declined = statistics.fmean(glyph_rate)
    beats_standard = gap > sd
    quiet_when_ignorant = untrained < 0.25
    knows_noise = noise_declined >= 3.0 * max(glyph_declined, 1e-9)
    passes = beats_standard and quiet_when_ignorant and knows_noise

    report = EntropyReport(
        passes=passes, seeds=seeds,
        accuracy=statistics.fmean(accuracies),
        risk_entropy=statistics.fmean(risks_e),
        risk_confidence=statistics.fmean(risks_c),
        risk_gap=gap, risk_sd=sd, untrained_bits=untrained,
        trained_bits=statistics.fmean(trained_bits),
        floor=statistics.fmean(floors),
        coverage=statistics.fmean(coverages),
        accepted_error=statistics.fmean(accepted),
        declined_all=statistics.fmean(all_error),
        noise_declined=noise_declined, glyph_declined=glyph_declined)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {seeds} seeds, accuracy "
        f"{report.accuracy:.4f}; at {coverage:.0%} coverage the error rate "
        f"is {report.risk_entropy:.4f} ranked by entropy against "
        f"{report.risk_confidence:.4f} by max-softmax "
        f"(gap {gap:+.4f} at seed sd {sd:.4f}, "
        f"{'entropy wins' if beats_standard else 'NOT a win'}); untrained "
        f"evidence {untrained:.3f} of 3.00 bits "
        f"({'quiet when ignorant' if quiet_when_ignorant else 'TOO LOUD'}); "
        f"floor {report.floor:.3f} bits keeps {report.coverage:.1%} of the "
        f"test split at {report.accepted_error:.4f} error against "
        f"{report.declined_all:.4f} for everything; noise declined "
        f"{noise_declined:.1%} against {glyph_declined:.1%} for glyphs "
        f"({'noise is caught' if knows_noise else 'NOISE IS NOT CAUGHT'})")
    return report


def rankers_compared(seeds: int = 5, epochs: int = 30,
                    coverage: float = 0.80) -> dict:
    """The same predictions ranked three ways, and two floors compared.

    Criterion 2 asked only whether entropy beat max-softmax. It did
    not, so the question became which measure is good for WHAT - and
    that needed margin in the comparison too. Reported rather than
    gated, because it was run after the criteria were fixed and
    reporting it as a pass would be scoring the coin after the
    throw.
    """
    root = Path(tempfile.mkdtemp(prefix="uq_rankers_"))
    scores = {
        "entropy": lambda p: p.entropy,
        "max-softmax": lambda p: -p.probability,
        "margin": lambda p: -p.margin,
    }
    risks: dict = {name: [] for name in scores}
    floors: dict = {"evidence": [], "margin": []}
    try:
        for seed in range(seeds):
            recognizer, _untrained, fit, test = _measure(seed, root, epochs)
            wrong = [w for _p, w in test]
            for name, score in scores.items():
                risks[name].append(risk_at_coverage(
                    [score(p) for p, _w in test], wrong, coverage))
            noise = [recognizer.net.predict_with_uncertainty(
                recognizer.features(x))
                for x in _noise_inputs(96, seed=1300 + seed)]
            for name, key in (("evidence", lambda p: p.evidence),
                              ("margin", lambda p: p.margin)):
                values = sorted(key(p) for p, _w in fit)
                floor = values[min(len(values) - 1,
                                   int(round(len(values)
                                             * (1.0 - coverage))))]
                kept = [(p, w) for p, w in test if key(p) >= floor]
                floors[name].append((
                    len(kept) / len(test),
                    sum(w for _p, w in kept) / len(kept) if kept else 0.0,
                    sum(1 for p in noise if key(p) < floor) / len(noise)))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return {
        "risk": {name: statistics.fmean(values)
                 for name, values in risks.items()},
        "floors": {name: tuple(statistics.fmean(column)
                               for column in zip(*rows))
                   for name, rows in floors.items()},
    }


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"seeds                {result.seeds}")
    print(f"accuracy             {result.accuracy:.4f}")
    print(f"untrained evidence   {result.untrained_bits:.3f} of 3.00 bits")
    print(f"trained evidence     {result.trained_bits:.3f} of 3.00 bits")
    print()
    print(f"error at 80% coverage, by entropy     {result.risk_entropy:.4f}")
    print(f"error at 80% coverage, by max-softmax {result.risk_confidence:.4f}")
    print(f"gap {result.risk_gap:+.4f} at seed sd {result.risk_sd:.4f}")
    print()
    print(f"floor                {result.floor:.3f} bits")
    print(f"coverage             {result.coverage:.1%}")
    print(f"error, accepted      {result.accepted_error:.4f}")
    print(f"error, everything    {result.declined_all:.4f}")
    print(f"noise declined       {result.noise_declined:.1%}")
    print(f"glyphs declined      {result.glyph_declined:.1%}")
    print()
    print(result.reason)
