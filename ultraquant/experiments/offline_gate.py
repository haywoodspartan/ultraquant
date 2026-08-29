"""The suggester, with the network unplugged. The offline adoption gate.

§11.39 adopted the embedding suggester on the live question path and
measured it wired: `build_session(semantic=True)` against the
identical session with it off. It passed, and it costs **a network
round trip on every question the lexical core cannot answer**.

§11.108 to §11.111 then established, in order, that the embedding's
edge is semantic rather than capacity, that it meets Stage 1's
criterion, that **58%** of it survives distillation into weights this
system owns, and that the vocabulary ceiling comes off with the
repository's own prose. **None of it was used by anything.** This
gate is that work reaching the question path.

**The three criteria are §11.39's, verbatim**, because the question
is whether the distilled encoder can hold the same job - not whether
it can pass an easier exam written for it. Two more are added for
what is genuinely different about an offline copy.

**The floor is the trap and it is handled explicitly.**
`COSINE_FLOOR = 0.75` was measured for the LIVE embedding. The
distilled encoder regresses *projected* teacher vectors, so its
cosines live on a different scale and 0.75 means something else
there. Inheriting it would be measuring a threshold, not a
representation. It is therefore **fitted on worlds the test arm never
sees** - the same discipline §11.102 used for its evidence floor.

**The criteria, written before the run.**

1. **The wall moves** - synonym-family assertive accuracy in the ON
   arm must beat the OFF arm's by more than the seed-to-seed sd.
2. **Decoys stay dead** - ghost and wrong-entity decoys must fall in
   NEITHER arm. One fall is a fail, sd or no sd. A distilled encoder
   that fabricates is worse than no encoder.
3. **Everything else is untouched** - canonical and reorder accuracy
   identical between arms. The suggester runs only where the lexical
   core could not assert, and this catches it leaking.
4. **The floor is fitted, not inherited** - chosen on fit worlds and
   reported on disjoint test worlds. A floor tuned and reported on
   the same worlds is a threshold that has seen its own answers.
5. **Nothing touches the network during evaluation** - the teacher is
   paid once, before any world is built, and the evaluation runs with
   no client in reach.

**PASSED — the distilled encoder holds the job, and one criterion
fitted something that turned out not to matter.**

| | off | on |
|---|---:|---:|
| synonym accuracy | 0.000 | **0.625** |
| untouched | 1.000 | 1.000 |
| decoy falls | 0 | 0 |

Gain **+0.625 at seed sd 0.484** - 1.29x the noise, which clears the
bar thinly and for a reason §11.39 already recorded: about 35% of
worlds ask a CHAINED synonym, which no embedding can reach and both
arms honestly miss. That is the mechanism's ceiling showing up as
variance, not a defect in this copy.

**Distilled in 1332 seconds over 216 teacher calls, none of them
during evaluation.** The suggester cannot tell it is not talking to
LM Studio.

**Criterion 4 was satisfied and pointless, which is worth more than a
clean pass.** The floor was fitted on six worlds and reported on
eight disjoint ones - and every candidate from 0.50 to 0.95 gave
**identical** gain and identical zero falls. The threshold never
binds.

Measured rather than guessed at, the reason is not that the cosines
collapsed: over fifteen probe pairs they span **0.41 to 0.97, spread
0.55**. They are systematically *shifted*, reading 0.74 / 0.87 / 0.71
where the teacher reads 0.53 / 0.72 / 0.52 on the same three pairs -
about 0.2 high, because a bag-of-words encoder over a small closed
vocabulary places texts that share words very close together.

So the correct matches score above every candidate floor and the
wrong ones are already gone: **the anchor rule and §11.36's
positional rule are doing all of the filtering, and the cosine
threshold none of it.** That is safe here and it is fragile in a way
worth writing down - a floor that never binds is not protection, it
is an untested assumption wearing protection's name. On a wider
vocabulary, where texts stop sharing words, it would begin to bind
and its value would be unmeasured.

**What this establishes.** §11.108 to §11.111 measured a shared
representation into existence; this is the first unit where the
system *uses* one. The semantic route on the live question path no
longer requires a network.

Run it::

    python -m ultraquant.experiments.offline_gate

Requires LM Studio with an embedding model for the distillation step
only. Without one the gate reports that it did not run.
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.experiments.semantic_gate import (_ENTITIES, _GHOSTS,
                                                  _MATERIALS, _PROPERTIES,
                                                  _SYNONYMS, _assertive)
from ultraquant.model.distilled import DistilledEmbedder
from ultraquant.model.prose import sentences as prose_sentences

__all__ = ["OfflineReport", "run_gate", "corpus_for_worlds", "FLOORS"]

#: Floors swept on the fit worlds. Wide, because the distilled scale
#: is not the teacher's and guessing near 0.75 would be assuming the
#: answer this gate exists to find.
FLOORS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def corpus_for_worlds(extra_prose: int = 100) -> list:
    """Every string the encoder could be asked about, plus prose.

    The world vocabulary is small and closed - six entities, three
    attributes and their synonyms, four materials, two properties,
    three ghosts - so it can be enumerated rather than sampled. The
    prose is §11.111's finding: a hundred sentences of ordinary
    English removed the vocabulary penalty entirely.

    The GHOSTS are included deliberately. Leaving them out would give
    the encoder no vector for exactly the words the decoys use, and a
    decoy that fails because the encoder has never heard the word is
    not a decoy that was rejected - it is one that was never offered.
    """
    texts: list = []
    subjects = list(_ENTITIES) + list(_GHOSTS)
    for subject in subjects:
        for attr, syn in _SYNONYMS.items():
            texts.append(f"the {subject} {attr}")
            texts.append(f"how {syn} is the {subject}?")
            texts.append(f"what is the {attr} of the {subject}?")
        texts.append(f"the {subject} material")
        for prop, prop_syn in _PROPERTIES.items():
            texts.append(f"how {prop_syn} is the {subject}?")
    for material in _MATERIALS:
        for prop in _PROPERTIES:
            texts.append(f"the {material} {prop}")
    texts += prose_sentences()[:extra_prose]
    return sorted(set(texts))


def _world(seed: int) -> dict:
    """One synonym world, built exactly as §11.39 builds them."""
    rng = random.Random(seed)
    entity, other = rng.sample(_ENTITIES, 2)
    attr, syn = rng.choice(list(_SYNONYMS.items()))
    ghost = rng.choice(_GHOSTS)
    value = rng.randrange(50, 950)
    other_value = rng.randrange(1000, 1999)
    facts = [(f"the {entity} {attr}", f"{value} meters"),
             (f"the {other} {attr}", f"{other_value} meters")]
    if rng.random() < 0.35:
        prop, prop_syn = rng.choice(list(_PROPERTIES.items()))
        material = rng.choice(_MATERIALS)
        chain = rng.randrange(1000, 9999)
        facts += [(f"the {entity} material", material),
                  (f"the {material} {prop}", f"{chain} units")]
        synonyms = [(f"how {prop_syn} is the {entity}?", str(chain))]
    else:
        synonyms = [(f"how {syn} is the {entity}?", str(value))]
    return {
        "facts": facts,
        "synonyms": synonyms,
        "untouched": [(f"what is the {entity} {attr}?", str(value)),
                      (f"what is the {attr} of the {other}?",
                       str(other_value))],
        "decoys": [f"how {syn} is the {ghost}?",
                   f"what is the {attr} of the {ghost}?"],
        "seed": seed,
    }


def _run_world(world: dict, semantic) -> tuple:
    """One world through one arm. Returns (synonym, untouched, falls)."""
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_offline_"))
    try:
        session = build_session(root, seed=world["seed"], semantic=semantic)
        for key, value in world["facts"]:
            run_pipeline(f"{key} is {value}", session)
        synonym = sum(
            int(_assertive(run_pipeline(q, session)[0])
                and want in run_pipeline(q, session)[0])
            for q, want in world["synonyms"]) / len(world["synonyms"])
        untouched = sum(
            int(want in run_pipeline(q, session)[0])
            for q, want in world["untouched"]) / len(world["untouched"])
        falls = sum(int(_assertive(run_pipeline(q, session)[0]))
                    for q in world["decoys"])
        return synonym, untouched, falls
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class OfflineReport:
    """Whether owned weights can hold the borrowed one's job.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ran: False when no teacher was reachable to distil from.
        model: The teacher distilled from, once.
        floor: The cosine floor fitted on the fit worlds.
        fit_worlds: Worlds used to choose it.
        test_worlds: Disjoint worlds everything else is reported on.
        synonym_off: Synonym accuracy, suggester off.
        synonym_on: The same with the distilled suggester on.
        synonym_sd: Seed sd of the per-world difference.
        untouched_off: Canonical and reorder accuracy, off.
        untouched_on: The same, on.
        falls_off: Decoy assertions with the suggester off.
        falls_on: Decoy assertions with it on.
        sweep: floor -> (synonym gain, decoy falls) on the fit worlds.
        distil_seconds: Seconds to distil, once.
        teacher_calls: Teacher calls made, all before evaluation.
        reason: Plain-language verdict.
    """

    passes: bool
    ran: bool = False
    model: str = ""
    floor: float = 0.0
    fit_worlds: int = 0
    test_worlds: int = 0
    synonym_off: float = 0.0
    synonym_on: float = 0.0
    synonym_sd: float = 0.0
    untouched_off: float = 0.0
    untouched_on: float = 0.0
    falls_off: int = 0
    falls_on: int = 0
    sweep: dict = field(default_factory=dict)
    distil_seconds: float = 0.0
    teacher_calls: int = 0
    reason: str = ""


def run_gate(fit_seeds: int = 6, test_seeds: int = 8,
             extra_prose: int = 100, epochs: int = 4000,
             model: str | None = None) -> OfflineReport:
    """Distil once, fit a floor, then measure on worlds it never saw."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    try:
        client = LMStudioClient()
        chosen = model or (client.embedding_models() or [None])[0]
        if chosen is None:
            raise RuntimeError("no embedding model in LM Studio")
        corpus = corpus_for_worlds(extra_prose)
        teacher = {text: client.embed([text], model=chosen)[0]
                   for text in corpus}
        started = time.perf_counter()
        embedder = DistilledEmbedder.distil(corpus, teacher, epochs=epochs)
        distil_seconds = time.perf_counter() - started
    except Exception as exc:                      # noqa: BLE001
        return OfflineReport(
            passes=False, ran=False,
            reason=f"COULD NOT RUN - {type(exc).__name__}: "
                   f"{str(exc)[:140]}. An adoption gate must not pass "
                   "with its mechanism absent.")

    # Criterion 5: the teacher is gone before a single world is built.
    teacher_calls = len(corpus)
    del client, teacher

    # Criterion 4: the floor is chosen here, on these worlds only.
    fit = [_world(seed) for seed in range(fit_seeds)]
    sweep: dict = {}
    for candidate in FLOORS:
        gains, falls = [], 0
        for world in fit:
            off_syn, _off_unt, off_falls = _run_world(world, False)
            suggester = _suggester(embedder, candidate)
            on_syn, _on_unt, on_falls = _run_world(world, suggester)
            gains.append(on_syn - off_syn)
            falls += on_falls + off_falls
        sweep[candidate] = (statistics.fmean(gains), falls)
    # A floor that lets a decoy through is not a candidate at any gain.
    clean = {f: g for f, (g, falls) in sweep.items() if falls == 0}
    floor = max(clean, key=clean.get) if clean else max(FLOORS)

    test = [_world(1000 + seed) for seed in range(test_seeds)]
    off_syn, on_syn, off_unt, on_unt = [], [], [], []
    falls_off = falls_on = 0
    for world in test:
        a, b, c = _run_world(world, False)
        off_syn.append(a)
        off_unt.append(b)
        falls_off += c
        a, b, c = _run_world(world, _suggester(embedder, floor))
        on_syn.append(a)
        on_unt.append(b)
        falls_on += c

    differences = [b - a for a, b in zip(off_syn, on_syn)]
    sd = statistics.pstdev(differences) if len(differences) > 1 else 0.0
    moves = statistics.fmean(differences) > sd
    dead = falls_off == 0 and falls_on == 0
    untouched = statistics.fmean(off_unt) == statistics.fmean(on_unt)
    passes = moves and dead and untouched

    report = OfflineReport(
        passes=passes, ran=True, model=chosen, floor=floor,
        fit_worlds=fit_seeds, test_worlds=test_seeds,
        synonym_off=statistics.fmean(off_syn),
        synonym_on=statistics.fmean(on_syn), synonym_sd=sd,
        untouched_off=statistics.fmean(off_unt),
        untouched_on=statistics.fmean(on_unt), falls_off=falls_off,
        falls_on=falls_on, sweep=sweep, distil_seconds=distil_seconds,
        teacher_calls=teacher_calls)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - distilled from {chosen} in "
        f"{distil_seconds:.0f}s over {teacher_calls} teacher calls, none "
        f"during evaluation; floor {floor:.2f} fitted on {fit_seeds} "
        f"worlds and reported on {test_seeds} disjoint ones; synonym "
        f"accuracy {report.synonym_off:.3f} off against "
        f"{report.synonym_on:.3f} on, a gain of "
        f"{report.synonym_on - report.synonym_off:+.3f} at seed sd "
        f"{sd:.3f} " + ("(the wall moves)" if moves
                        else "- INSIDE THE NOISE, the wall did not move")
        + f"; decoys fell {falls_off} off and {falls_on} on "
        + ("(dead in both)" if dead else "- A DECOY FELL")
        + f"; untouched {report.untouched_off:.3f} against "
        f"{report.untouched_on:.3f} "
        + ("(identical)" if untouched else "- THE SUGGESTER LEAKED"))
    return report


def _suggester(embedder, floor: float):
    from ultraquant.reason.semantic import SemanticSuggester

    return SemanticSuggester(embedder=embedder, floor=floor)


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"teacher: {result.model}   distilled in "
          f"{result.distil_seconds:.0f}s over {result.teacher_calls} calls")
    print()
    print(f"{'floor':>7}{'synonym gain':>15}{'decoy falls':>14}")
    for candidate in FLOORS:
        gain, falls = result.sweep[candidate]
        mark = "  <- chosen" if candidate == result.floor else ""
        print(f"{candidate:>7.2f}{gain:>+15.3f}{falls:>14d}{mark}")
    print()
    print(f"{'':<12}{'off':>9}{'on':>9}")
    print(f"{'synonym':<12}{result.synonym_off:>9.3f}"
          f"{result.synonym_on:>9.3f}")
    print(f"{'untouched':<12}{result.untouched_off:>9.3f}"
          f"{result.untouched_on:>9.3f}")
    print(f"{'decoy falls':<12}{result.falls_off:>9d}{result.falls_on:>9d}")
    print()
    print(result.reason)
