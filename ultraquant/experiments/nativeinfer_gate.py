"""Two spreads that converge the same way. The native inference gate.

The hardest thing in this system to port faithfully, because its
correctness lives in ORDER as much as in arithmetic: which facts the
index surfaces for which probe, which lineage reaches a node first,
which of several equally-strong convergences wins. Python dicts
preserve insertion order and the spread leans on that, so the native
tier uses insertion-ordered containers wherever the Python side walks
a dict - a std::map would have sorted them and quietly changed which
chain answered.

The activation strengths are powers of one half, exact in binary
floating point, so the one place a port would normally drift it
cannot. Everything else that could differ is a rule: whole-value
bridging, the single-origin lineage, origin coverage at one absent
token, the terminal relation slot, the adjacent-bigram order rule,
and the anagram refusal.

This gate also closes §11.91's named debt. The curiosity hint the
native pipeline could not produce came from `missing_premise`, which
is ported here, so the chat gate's 71 hint-only differences should
now be zero - and that is checked rather than assumed.

**The criteria, written before the run.**

1. **Parity is total** - every derivation, as its spoken sentence
   with premises and confidence, identical. One mismatch is a fail.
2. **Refusal is symmetric** - where Python derives nothing, the
   native tier derives nothing. A tier that converges MORE is worse
   than one that converges less: every extra answer is a fabrication
   the rule stack exists to refuse.
3. **Curiosity gaps match** - the minted premise key, the bridge it
   came through, and its value.
4. **The decoys are present and reported** - epithet subjects,
   anagram subjects, denial bridges, and modifier questions each
   appear in the corpus, with counts, so a green run cannot come
   from a world that quietly stopped being hard.

**It was run once, and PASSED - on the first attempt, which is not
what I expected of this one.**

| | |
|---|---:|
| questions put to both engines | 368 |
| derivations that differed | **0** |
| derived only by Python | 0 |
| derived only by the native tier | 0 |
| curiosity gaps that differed | **0** |

Across forty worlds: 130 two-fact chains, 59 unreachable questions,
36 epithet decoys, 30 three-fact chains, 30 four-fact chains, 30
denial bridges, 28 anagram decoys, 25 modifier questions.

The refusal-symmetry line is the one that matters most and it held in
both directions: **zero questions were derived only by the native
tier.** Every epithet decoy, every anagram subject and every denial
bridge that Python refuses, the native tier refuses too. A port that
converged more often would have looked better on a correctness
score and been strictly worse, because each extra answer is a pun
the rule stack exists to refuse.

That it passed first time is worth attributing rather than
celebrating: the rules were not re-derived here, they were copied,
and the two places a C++ port would ordinarily drift had already
been closed - insertion-ordered containers where the Python side
walks a dict, and activation strengths that are powers of one half
and therefore exact in binary.

Run it::

    python -m ultraquant.experiments.nativeinfer_gate
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["InferParityReport", "run_gate", "probe_path"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_infer.exe"


def probe_path() -> Path:
    """Where the built engine lives, so tests can skip without it."""
    return _PROBE


@dataclass
class InferParityReport:
    """Parity over derivations and gaps, plus decoy coverage.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        asked: Questions put to both engines.
        mismatches: Derivations that differed.
        native_only: Questions the native tier answered and Python did not.
        python_only: Questions Python answered and the native tier did not.
        gap_mismatches: Curiosity gaps that differed.
        kinds: Question kind -> how many were asked.
        first_mismatch: The first disagreement, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    asked: int = 0
    mismatches: int = 0
    native_only: int = 0
    python_only: int = 0
    gap_mismatches: int = 0
    kinds: dict = field(default_factory=dict)
    first_mismatch: str = ""
    reason: str = ""


_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great", "little", "grand"]
_MIDDLES = ["stone", "river", "hill", "market", "abbey", "castle",
            "harbour", "garden", "forest", "valley"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall", "barn", "arch"]
_MATERIALS = ["iron", "steel", "copper", "granite", "oak", "bronze"]
_PEOPLE = ["wren", "soane", "nash", "barry", "pugin"]
_PLACES = ["bristol", "norwich", "durham", "exeter"]
_REGIONS = ["west", "east", "north", "south"]


def _world(rng: random.Random):
    """Statements and questions: chains, decoys, denials, modifiers."""
    def entity() -> str:
        return (f"{rng.choice(_PREFIXES)} {rng.choice(_MIDDLES)} "
                f"{rng.choice(_NOUNS)}")

    statements: list[tuple[str, str, bool]] = []
    questions: list[tuple[str, str]] = []
    used: set[str] = set()

    def name() -> str:
        for _ in range(40):
            candidate = entity()
            if candidate not in used:
                used.add(candidate)
                return candidate
        return entity()

    # Two-fact chains: subject -> material -> property.
    for _ in range(rng.choice((2, 3))):
        subject = name()
        material = rng.choice(_MATERIALS)
        attribute = rng.choice(["hardness", "conductivity", "density"])
        statements.append((f"{subject} material", material, False))
        statements.append((f"{material} {attribute}",
                           f"{rng.randint(100, 900)} units", False))
        questions.append((f"what is the {subject} {attribute}?", "chain2"))

    # Longer chains: subject -> architect -> hometown -> region -> value.
    if rng.random() < 0.8:
        subject = name()
        person = rng.choice(_PEOPLE)
        place = rng.choice(_PLACES)
        region = rng.choice(_REGIONS)
        statements.append((f"{subject} architect", person, False))
        statements.append((f"{person} hometown", place, False))
        statements.append((f"{place} region", region, False))
        statements.append((f"{region} climate", "temperate", False))
        questions.append((f"what is the {subject} architect hometown?",
                          "chain2"))
        questions.append((f"what is the {subject} architect hometown "
                          f"region?", "chain3"))
        questions.append((f"what is the {subject} architect hometown "
                          f"region climate?", "chain4"))

    # An epithet decoy: a longer subject that must not answer for the
    # shorter one, nor the shorter for the longer.
    if rng.random() < 0.7:
        base = rng.choice(_PEOPLE)
        epithet = f"{base} the younger"
        statements.append((f"{epithet} hometown", rng.choice(_PLACES),
                           False))
        questions.append((f"what is the {base} hometown?", "epithet"))

    # An anagram decoy: same tokens, different entity.
    if rng.random() < 0.7:
        a, b, c = rng.choice(_PREFIXES), rng.choice(_MIDDLES), \
            rng.choice(_NOUNS)
        statements.append((f"{a} {b} {c} material", rng.choice(_MATERIALS),
                           False))
        questions.append((f"what is the {a} {c} {b} hardness?", "anagram"))

    # A denial that must never bridge.
    if rng.random() < 0.8:
        subject = name()
        material = rng.choice(_MATERIALS)
        statements.append((f"{subject} material", material, True))
        statements.append((f"{material} hardness",
                           f"{rng.randint(100, 900)} units", False))
        questions.append((f"what is the {subject} hardness?", "denial"))

    # A modifier question: one library-unknown adjective in front.
    if rng.random() < 0.7 and statements:
        subject = rng.choice([key.rsplit(" ", 1)[0]
                              for key, _v, neg in statements
                              if key.endswith(" material") and not neg]
                             or ["north stone tower"])
        questions.append((f"what is the weathered {subject} hardness?",
                          "modifier"))

    # Questions with nothing to reach, for the refusal symmetry line.
    for _ in range(rng.choice((1, 2))):
        questions.append((f"what is the {entity()} conductivity?",
                          "unreachable"))

    rng.shuffle(questions)
    return statements, questions


def _python_answers(statements, questions):
    """What the tier that defines the semantics derives."""
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.reason.inference import infer, missing_premise

    memory = SystematicMemory()
    for key, value, negated in statements:
        memory.remember_fact(key, value, 0.6, negated)
    out: list[str] = []
    for question, _kind in questions:
        result = infer(question, memory)
        out.append("infer|" + (result.describe() if result else "none"))
        gap = missing_premise(question, memory)
        out.append("gap|" + (f"{gap['premise_key']}|{gap['via_key']}|"
                             f"{gap['via_value']}" if gap else "none"))
    return out


def _native_answers(statements, questions):
    """The same world and questions, put to the native engine."""
    lines = [f"remember 0.6 {1 if negated else 0} {key}|{value}"
             for key, value, negated in statements]
    for question, _kind in questions:
        lines.append(f"infer {question}")
        lines.append(f"gap {question}")
    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    answers = got.stdout.splitlines()
    return [line for line in answers if line != "ok"]


def run_gate(worlds: int = 40, seed: int = 0) -> InferParityReport:
    """Build the same worlds twice and compare every derivation."""
    if not _PROBE.exists():
        return InferParityReport(
            passes=False,
            reason="FAIL - the native engine is not built; run "
                   "native/uq/build.ps1")
    asked = 0
    mismatches = 0
    native_only = 0
    python_only = 0
    gap_mismatches = 0
    kinds: dict = {}
    first = ""
    for index in range(worlds):
        rng = random.Random(51000 + seed * 100 + index)
        statements, questions = _world(rng)
        for _question, kind in questions:
            kinds[kind] = kinds.get(kind, 0) + 1
        want = _python_answers(statements, questions)
        have = _native_answers(statements, questions)
        for step, (question, kind) in enumerate(questions):
            asked += 1
            want_infer = want[step * 2]
            want_gap = want[step * 2 + 1]
            have_infer = (have[step * 2] if step * 2 < len(have)
                          else "(no answer)")
            have_gap = (have[step * 2 + 1] if step * 2 + 1 < len(have)
                        else "(no answer)")
            if want_infer != have_infer:
                mismatches += 1
                if want_infer == "infer|none":
                    native_only += 1
                elif have_infer == "infer|none":
                    python_only += 1
                if not first:
                    first = (f"world {index} [{kind}] {question!r}\n"
                             f"    python: {want_infer}\n"
                             f"    native: {have_infer}")
            if want_gap != have_gap:
                gap_mismatches += 1
                if not first:
                    first = (f"world {index} [{kind}] {question!r}\n"
                             f"    python gap: {want_gap}\n"
                             f"    native gap: {have_gap}")
    passes = mismatches == 0 and gap_mismatches == 0
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {asked} questions put to "
        f"both engines across {worlds} worlds, {mismatches} "
        f"derivations differed ({python_only} derived only by Python, "
        f"{native_only} only by the native tier), {gap_mismatches} "
        f"curiosity gaps differed"
        + (f"\nfirst:\n{first}" if first else ""))
    return InferParityReport(passes=passes, asked=asked,
                             mismatches=mismatches,
                             native_only=native_only,
                             python_only=python_only,
                             gap_mismatches=gap_mismatches, kinds=kinds,
                             first_mismatch=first, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"questions put to both engines: {report.asked}")
    print(f"derivations that differed: {report.mismatches}")
    print(f"  derived only by Python: {report.python_only}")
    print(f"  derived only by the native tier: {report.native_only}")
    print(f"curiosity gaps that differed: {report.gap_mismatches}")
    print()
    for kind, count in sorted(report.kinds.items()):
        print(f"  {kind:<14} {count:5d} questions")
    print()
    print(report.reason)
