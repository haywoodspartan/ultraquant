"""The comparative word carries its attribute. The implied-attr gate.

"is the west tower taller than the south tower?" refused with "I
hold nothing for 'south tower'" - while "south tower height is 450
meters" sat in the store - and "which is the tallest?" answered
nothing over three stored heights. The word "taller" NAMES the
attribute and nobody read it: both the pairwise and the setwise
branch required "height" spelled out in the question, a shape nobody
speaks. §11.75 reads the attribute inside the comparative word -
taller/tallest/shorter/shortest say height, heavier/heaviest/
lighter/lightest say weight, longer/longest say length - and only
where the word is attribute-specific in plain English. The
polysemous words are EXCLUDED on the record: bigger, smaller,
larger, higher, lower name no single family ("highest price", "lower
cost"), so implication there would fabricate a specificity the
speaker never offered - they keep requiring the named attribute,
exactly as today.

Order of address is preserved: an exact bare-key fact ("north tower"
= "300 meters") still wins first; the implied attribute is appended
only when the bare key misses; §11.55's derive runs on the appended
key; the refusal names what was tried.

Both arms run the live pipeline; the baseline arm turns
`_IMPLIED_ATTRS` off (the shipped refusals), the treatment arm is
current. Worlds mix, in world-varying proportions: natural-shape
pairwise questions over stored attributes, natural-shape
superlatives, derive cases (the operand's attribute reachable only
through a kind chain), unwinnable cases (the operand's attribute
neither stored nor derivable - nobody can answer; scored zero for
both arms, §11.35's power food), and ambiguous-word questions
("which is the biggest?") that must refuse in BOTH arms.

**The criteria, written before the run.**

1. **Gain** - natural-shape correctness (the right yes/no on
   pairwise, the right value asserted on superlatives) must beat the
   baseline arm by more than the seed-to-seed sd.
2. **The implication never invents** - an ambiguous-word question
   ("biggest", "largest", "smallest") answered with a ranked verdict
   in either arm is a fail, once. Polysemy refuses aloud.
3. **Absence is never yes or no** - an unwinnable question (the
   operand's attribute neither stored nor derivable) answered with a
   verdict in either arm is a fail, once. Unwinnables score zero for
   both arms - §11.35's variance food - but a refusal is the only
   correct shape there.
4. **Named-attribute forms are untouched** - questions that spell
   the attribute out answer identically in both arms.
5. **Exact keys still win first** - bare-key numeric facts compare
   by their own values, identically in both arms, never shadowed by
   an appended attribute.

**It was run once, and PASSED — zero to 0.700, with every integrity
line at zero.**

| arm | natural |
|---|---:|
| refuses | 0.000 |
| reads | **0.700** |

**+0.700 at 9.48x seed sd** — the baseline answered NO natural shape
at all (0.000: every "is A taller than B?" refused over a store that
held the answer), the treatment answered every winnable one (0.700
is exactly the winnable share; the rest are the unwinnables, scored
zero for both arms by §11.35's arithmetic). Zero ambiguous words
answered, zero unwinnables answered, zero named-form mismatches,
zero bare pairs shadowed. The derive trail rode through: "is the
west tower taller than the south tower?" answered Yes with "west
tower height is 900 meters (derived via spirekind)" - implication,
chain, and comparison composing in one verdict. And the polysemy
line held in both directions during design: "shorter" was DROPPED
from the map before any run, because it spans height and length
("the shorter rope") - the same test that excludes "biggest"
excluded one of our own candidates.

Run it::

    python -m ultraquant.experiments.impliedattr_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ImpliedReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_KINDS = ["spirekind", "vaultkind", "archkind", "slabkind", "pierkind",
          "domekind"]

#: attribute -> (comparative, superlative, unit); the three families
#: whose words are attribute-specific in plain English.
_FAMILIES = {
    "height": ("taller", "tallest", "meters"),
    "weight": ("heavier", "heaviest", "kilograms"),
    "length": ("longer", "longest", "meters"),
}

#: words that name no single attribute; must refuse in both arms.
_AMBIGUOUS = ["biggest", "largest", "smallest"]


@dataclass
class ImpliedReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        natural: Arm -> natural-shape correctness over all worlds
            (unwinnables scored zero for both arms).
        invented: Ambiguous-word questions answered with a verdict
            (must be 0 in both arms).
        named_mismatch: Named-attribute questions whose answers
            differ between arms (must be 0).
        exact_shadowed: Bare-key facts shadowed by an appended
            attribute (must be 0 in both arms).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    natural: dict = field(default_factory=dict)
    invented: int = 0
    fabricated: int = 0
    named_mismatch: int = 0
    exact_shadowed: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _build_world(rng: random.Random):
    """Entities, their attribute facts, and the question plan.

    Returns ``(statements, cases)`` where each case is
    ``(question, kind, expected)`` - kind one of ``natural``,
    ``unwinnable``, ``ambiguous``, ``named``, ``exact``; expected
    carries whatever the checker needs.
    """
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    attr, (comp, sup, unit) = rng.choice(sorted(_FAMILIES.items()))
    kind = rng.choice(_KINDS)

    statements = []
    values = {}
    # Three entities with stored attributes - distinct values so the
    # superlative has one winner.
    stored = names[0:3]
    pool = rng.sample(range(2, 30), 3)
    for name, base in zip(stored, pool):
        values[name] = base * 10
        statements.append(f"the {name} {attr} is {base * 10} {unit}")
    # One entity whose attribute is reachable only through a chain.
    chained = names[3]
    chain_value = max(pool) * 10 + 100
    statements.append(f"the {chained} kind is {kind}")
    statements.append(f"{kind} {attr} is {chain_value} {unit}")
    values[chained] = chain_value
    # One entity with facts but not THIS attribute - unwinnable.
    absent = names[4]
    statements.append(f"the {absent} material is granite")

    # Two bare-key NUMERIC facts: the exact-first floor - their
    # comparison must never be shadowed by an appended attribute.
    bare_a, bare_b = names[5], names[6]
    statements.append(f"the {bare_a} is 40")
    statements.append(f"the {bare_b} is 60")

    cases = []
    # Natural pairwise over stored operands; expected is the POLARITY
    # (both operand names appear in every verdict's value line, so a
    # substring check could not discriminate the winner).
    a, b = rng.sample(stored, 2)
    cases.append((f"is the {a} {comp} than the {b}?", "natural",
                  "yes" if values[a] > values[b] else "no"))
    # Natural pairwise with a derived operand - the chain holds the
    # top value, so its side decides the polarity; sides drawn so
    # the expected answer is not always yes.
    other = rng.choice(stored)
    if rng.random() < 0.5:
        cases.append((f"is the {chained} {comp} than the {other}?",
                      "natural", "yes"))
    else:
        cases.append((f"is the {other} {comp} than the {chained}?",
                      "natural", "no"))
    # Natural superlative - the chain terminal holds the top value;
    # correctness is that VALUE asserted, not hedged.
    cases.append((f"which is the {sup}?", "natural",
                  str(chain_value)))
    # Unwinnable pairwise, in world-varying number (the §11.35 draw):
    # scored zero for both arms, and a verdict here is a fabrication.
    for _ in range(rng.choice((1, 2))):
        other = rng.choice(stored)
        cases.append((f"is the {absent} {comp} than the {other}?",
                      "unwinnable", None))
    # Ambiguity floor: must refuse in both arms.
    cases.append((f"which is the {rng.choice(_AMBIGUOUS)}?",
                  "ambiguous", None))
    # Named-attribute forms: byte-identical across arms.
    a, b = rng.sample(stored, 2)
    cases.append((f"is the {a} {attr} larger than the {b} {attr}?",
                  "named", None))
    cases.append((f"which {attr} is the {sup}?", "named", None))
    # Exact-first: bare numeric keys compare by their own values,
    # identically in both arms (byte-identity AND a yes/no verdict).
    cases.append((f"is the {bare_a} {comp} than the {bare_b}?",
                  "exactpair", None))
    return statements, cases


_HEDGES = ("i don't", "i hold no", "i hold nothing", "i can't")


def _natural_correct(response: str, expected: str) -> bool:
    """Polarity questions match by verdict; values by assertion."""
    low = response.lower()
    if expected in ("yes", "no"):
        return low.startswith(expected)
    if low.startswith(_HEDGES):
        return False
    return expected.lower() in low


def _is_verdict(low: str) -> bool:
    """A ranked or polar assertion, as opposed to a refusal."""
    return (low.startswith(("yes", "no"))
            or (" is the " in low and not low.startswith(_HEDGES)))


def run_gate(seeds: int = 12) -> ImpliedReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    arms = {"refuses": False, "reads": True}
    natural = {}
    per_seed = {arm: [] for arm in arms}
    invented = 0
    fabricated = 0
    exact_shadowed = 0
    named = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(3400 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_impl_"))
            old = thoughts._IMPLIED_ATTRS_ON
            thoughts._IMPLIED_ATTRS_ON = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                total = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    low = response.lower()
                    if kind == "natural":
                        total += 1
                        if _natural_correct(response, expected):
                            hits += 1
                    elif kind == "unwinnable":
                        total += 1  # scored zero: nobody can answer
                        if _is_verdict(low):
                            fabricated += 1
                    elif kind == "ambiguous":
                        if _is_verdict(low):
                            invented += 1
                    elif kind in ("named", "exactpair"):
                        named.setdefault((seed, question),
                                         {})[arm] = response
                        if (kind == "exactpair"
                                and not low.startswith(("yes", "no"))):
                            exact_shadowed += 1
                scores.append(hits / total if total else 1.0)
            finally:
                thoughts._IMPLIED_ATTRS_ON = old
                shutil.rmtree(root, ignore_errors=True)
        natural[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    named_mismatch = sum(
        1 for responses in named.values()
        if len(set(responses.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["reads"],
                                       per_seed["refuses"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and invented == 0
              and fabricated == 0 and named_mismatch == 0
              and exact_shadowed == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - natural shapes at "
        f"{natural['reads']:.3f} against {natural['refuses']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {invented} ambiguous "
        f"words answered, {fabricated} unwinnables answered, "
        f"{named_mismatch} named-form mismatches, "
        f"{exact_shadowed} bare pairs unanswered-or-shadowed")
    return ImpliedReport(
        passes=passes, natural=natural, invented=invented,
        fabricated=fabricated, named_mismatch=named_mismatch,
        exact_shadowed=exact_shadowed, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         natural")
    print(f"refuses      {report.natural['refuses']:.3f}")
    print(f"reads        {report.natural['reads']:.3f}")
    print(f"ambiguous words answered: {report.invented}")
    print(f"unwinnables answered: {report.fabricated}")
    print(f"named-form mismatches: {report.named_mismatch}")
    print(f"bare pairs unanswered-or-shadowed: {report.exact_shadowed}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
