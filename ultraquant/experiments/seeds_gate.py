"""The dictionary tells the frontier what exists. The gate.

"We need it to ask its own questions." The frontier's questions came only
from gaps in what the library holds; the dictionary (§11.167) lists what
exists. Here an attribute's subjects, read through the dictionary, yield the
kinds they belong to - senses disambiguated by the unambiguous subjects -
and the frontier asks about the other members of those kinds: countries the
library never heard of. A word is not a sense: novel titles match fictional
characters, "Quebec" is a province and a city; coherence, disambiguation and
alias guards keep the questions honest.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 afa707f0...) before the implementation existed. The world: a copy of
the pinned library after §11.155's filing, with a lexicon built from the
tracked release; the first source's recorded teacher, answering UNKNOWN
(five samples) to anything it has no recording for.
1. **Exactly these targets**: ``seeds.dictionary_targets`` equals the exam's
   own reading of the design (125: 120 capitals, 5 atomic numbers).
2. **No lexicon, no change**: without a lexicon, pending and one study round
   equal the golden record taken with §11.169's code before any code for
   this unit (``records/seeds_11170_golden.json``).
3. **The frontier asks them**: rounds of the first source with the lexicon,
   until used up, ask every expected dictionary target exactly once, and no
   round is used up while one is unasked.
4. **The guards hold**: no author target; no target named ununbium,
   ununtrium, ununquadium, ununpentium or ununhexium.
5. **The exam can fail**: P121 (coherent always True) breaches 4; P122
   (core_kinds without disambiguation) breaches 1; P123 (is_alias always
   False) breaches 4; P124 (pending without the dictionary) breaches 3.
6. **Nothing regresses**: a sweep matches the ledger. Checked outside.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["SeedsReport", "run_gate"]

HERE = Path(__file__).with_name("records")
GOLDEN = HERE / "seeds_11170_golden.json"
ALIASES = {"ununbium", "ununtrium", "ununquadium", "ununpentium", "ununhexium"}
_RUN: dict = {}


# -- the exam's own reading of the design ------------------------------------------------

def _noun_senses(lexicon, subject):
    return [s["synset"] for s in lexicon.senses(subject) if s["pos"] == "n"]


def _direct(lexicon, synset_id):
    record = lexicon.synset(synset_id) or {}
    return list(record.get("instance_hypernym") or record.get("hypernym") or [])


def _ancestors(lexicon, synset_id):
    seen, order, level = {synset_id}, [], [synset_id]
    while level:
        following = []
        for sid in level:
            record = lexicon.synset(sid) or {}
            for parent in (*record.get("instance_hypernym", ()), *record.get("hypernym", ())):
                if parent not in seen:
                    seen.add(parent)
                    order.append(parent)
                    following.append(parent)
        level = following
    return order


def _expected(memory, stash, lexicon) -> list:
    """The design's targets, as Target objects, sorted by (attribute, subject)."""
    from ultraquant.distill import targets
    from ultraquant.lexicon import normalize_word
    from ultraquant.memory.factshards import normalize_subject
    facts = defaultdict(dict)
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        if record.get("subject") and record.get("attribute"):
            facts[record["attribute"]][record["subject"]] = str(record.get("value"))
    out = {}
    for attribute, held in sorted(facts.items()):
        n = len(held)
        senses = {s: _noun_senses(lexicon, s) for s in held}
        cover = defaultdict(set)
        for s, ss in senses.items():
            for x in ss:
                for k in _ancestors(lexicon, x):
                    cover[k].add(s)
        top = max((len(v) for v in cover.values()), default=0)
        form = targets.question_form(stash, attribute)
        if n == 0 or 2 * top < n or form is None:
            continue
        core = defaultdict(int)
        for s, ss in senses.items():
            if len(ss) == 1:
                for k in _direct(lexicon, ss[0]):
                    core[k] += 1
        support = dict(core)
        for s, ss in senses.items():
            if len(ss) > 1:
                chosen = next((x for x in ss if any(k in core for k in _direct(lexicon, x))), None)
                if chosen is not None:
                    for k in _direct(lexicon, chosen):
                        if k in core:
                            support[k] += 1
        kinds = sorted(k for k, c in support.items() if c >= 2)
        held_same = {normalize_subject(s) for s in held}
        a = normalize_word(attribute)
        held_values = {normalize_word(v) for v in held.values()}
        witnessed = sum(
            1 for s, v in held.items()
            if any(normalize_word(m) == normalize_word(f"{attribute} {v}")
                   for x in senses[s] for m in (lexicon.synset(x) or {}).get("members", [])))
        names = set()
        for k in kinds:
            record = lexicon.synset(k) or {}
            for child in record.get("instance", []):
                members = (lexicon.synset(child) or {}).get("members", [])
                if members and not any(normalize_subject(m) in held_same for m in members):
                    names.add(members[0])
            if 2 * witnessed >= n:
                for child in record.get("hyponym", []):
                    members = (lexicon.synset(child) or {}).get("members", [])
                    if not members or any(normalize_subject(m) in held_same for m in members):
                        continue
                    values = [normalize_word(m)[len(a) + 1:] for m in members
                              if normalize_word(m).startswith(a + " ")]
                    if values and not any(v in held_values for v in values):
                        names.add(members[0])
        category, question = form
        for name in names:
            out[attribute, name] = targets.Target(category, name, question.format(subject=name),
                                                  attribute)
    return [out[key] for key in sorted(out)]


def _triples(items) -> set:
    return {(t.attribute, t.subject, t.question) for t in items}


# -- the world ---------------------------------------------------------------------------

class _Answering:
    """The recorded first source; UNKNOWN (five samples) to anything it has no recording for."""

    def __init__(self, teacher):
        self.teacher, self.spec = teacher, teacher.spec

    def ask(self, questions, *, samples, **kwargs):
        known = [q for q in questions if q in self.teacher.recorded]
        answers = dict(zip(known, self.teacher.ask(known, samples=samples, **kwargs))) if known else {}
        return [answers.get(q, ["UNKNOWN"] * samples) for q in questions]


@contextlib.contextmanager
def _world(with_lexicon: bool = True):
    from ultraquant.distill import sources
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import session_gate as SES
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.shape_gate import _live_copy
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    try:
        with _live_copy() as lib:
            lexicon = None
            if with_lexicon:
                importlib.import_module("ultraquant.lexicon.build").build(lib.root)
                lexicon = importlib.import_module("ultraquant.lexicon.lexicon").Lexicon.open(lib.root)
            teacher = SES._factory()(TG.FIRST)
            ledger = sources.SourceLedger(lib.root / "sources.json")
            yield lib, lexicon, teacher, ledger, TG.FIRST
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes


# -- 1: exactly these targets ------------------------------------------------------------

def exact_targets() -> bool:
    seeds = importlib.import_module("ultraquant.distill.seeds")
    with _world() as (lib, lexicon, _teacher, _ledger, _first):
        got = _triples(seeds.dictionary_targets(lib.memory, lib.stash, lexicon))
        want = _triples(_expected(lib.memory, lib.stash, lexicon))
    _RUN["exact"] = {"got": len(got), "want": len(want), "extra": sorted(got - want)[:5],
                     "missing": sorted(want - got)[:5],
                     "by attribute": {a: sum(1 for t in want if t[0] == a) for a in {t[0] for t in want}}}
    return got == want and len(want) == 125


# -- 2: no lexicon, no change ------------------------------------------------------------

def no_change() -> bool:
    from ultraquant.distill import elicit, frontier
    from ultraquant.experiments.ownquestions_gate import _decided
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # pending may learn a kind as it goes (kind_of): each check gets its own world,
    # as the golden record did.
    with _world(with_lexicon=False) as (lib, _lexicon, teacher, ledger, first):
        plain = [elicit.question_id(t) for t in frontier.pending(lib.memory, lib.stash, teacher, ledger, first)]
    with _world(with_lexicon=False) as (lib, _lexicon, teacher, ledger, first):
        explicit = [elicit.question_id(t) for t in frontier.pending(
            lib.memory, lib.stash, teacher, ledger, first, lexicon=None)]
    with _world(with_lexicon=False) as (lib, _lexicon, teacher, ledger, first):
        scratch = Path(tempfile.mkdtemp(prefix="uq_seeds_"))
        result = frontier.study_round(lib.memory, lib.stash, teacher, ledger, first,
                                      confidence=_decided()[3]["wilson_lower"],
                                      run_id="seeds-golden-1", records_path=scratch / "1.jsonl",
                                      approver=lib.approver(), lexicon=None)
        rows = ledger._read()
    _RUN["no change"] = {"pending": plain, "round": result,
                         "same round": result == golden["round"], "same ledger": rows == golden["ledger"]}
    return (plain == golden["pending"] and explicit == golden["pending"]
            and result == golden["round"] and rows == golden["ledger"])


# -- 3: the frontier asks them -----------------------------------------------------------

def asks_them() -> bool:
    from ultraquant.distill import elicit, frontier
    from ultraquant.experiments.ownquestions_gate import _decided
    with _world() as (lib, lexicon, teacher, ledger, first):
        answering = _Answering(teacher)
        expected_ever, bad, rounds = set(), [], []
        for n in range(1, 13):
            before = {elicit.question_id(t) for t in _expected(lib.memory, lib.stash, lexicon)}
            expected_ever |= before
            scratch = Path(tempfile.mkdtemp(prefix="uq_seeds_"))
            result = frontier.study_round(lib.memory, lib.stash, answering, ledger, first,
                                          confidence=_decided()[3]["wilson_lower"],
                                          run_id=f"seeds-{n}", records_path=scratch / f"{n}.jsonl",
                                          approver=lib.approver(), lexicon=lexicon)
            asked = [row["question_id"] for row in ledger._read().get(first, [])]
            after = {elicit.question_id(t) for t in _expected(lib.memory, lib.stash, lexicon)}
            unasked = sorted(after - set(asked))
            rounds.append({"asked": result["asked"], "used_up": result["used_up"], "unasked": len(unasked)})
            if result["used_up"] and unasked:
                bad.append(("used up with targets unasked", n, unasked[:3]))
            if result["used_up"]:
                break
        asked = [row["question_id"] for row in ledger._read().get(first, [])]
        counts = {qid: asked.count(qid) for qid in expected_ever}
        not_once = sorted(q for q, c in counts.items() if c != 1)
    _RUN["asks"] = {"rounds": rounds, "expected ever": len(expected_ever), "not once": not_once[:5],
                    "bad": bad[:3]}
    return (bool(expected_ever) and not not_once and not bad
            and bool(rounds) and rounds[-1]["used_up"])


# -- 4: the guards hold ------------------------------------------------------------------

def guards() -> bool:
    seeds = importlib.import_module("ultraquant.distill.seeds")
    with _world() as (lib, lexicon, _teacher, _ledger, _first):
        got = seeds.dictionary_targets(lib.memory, lib.stash, lexicon)
    authors = [t.subject for t in got if t.attribute == "author"]
    aliases = [t.subject for t in got if t.subject.lower() in ALIASES]
    _RUN["guards"] = {"author targets": authors[:5], "alias targets": aliases}
    return not authors and not aliases


CASES = {"1 exact targets": exact_targets, "2 no change": no_change,
         "3 asks them": asks_them, "4 guards": guards}


# -- plants --------------------------------------------------------------------------------

def _plants():
    seeds = importlib.import_module("ultraquant.distill.seeds")
    from ultraquant.distill import frontier

    def every_sense(lexicon, subjects):
        counts = defaultdict(int)
        for subject in subjects:
            for sid in _noun_senses(lexicon, subject):
                for kind in _direct(lexicon, sid):
                    counts[kind] += 1
        return {k: c for k, c in counts.items() if c >= 2}

    real_pending = frontier.pending

    def no_dictionary(memory, stash, teacher, ledger, source, lexicon=None):
        return real_pending(memory, stash, teacher, ledger, source)

    return [
        ("P121 coherent always True", "4 guards",
         [mock.patch.object(seeds, "coherent", lambda *a, **k: True)]),
        ("P122 core_kinds without disambiguation", "1 exact targets",
         [mock.patch.object(seeds, "core_kinds", every_sense)]),
        ("P123 is_alias always False", "4 guards",
         [mock.patch.object(seeds, "is_alias", lambda *a, **k: False)]),
        ("P124 pending without the dictionary", "3 asks them",
         [mock.patch.object(frontier, "pending", no_dictionary)]),
    ]


@dataclass
class SeedsReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> SeedsReport:
    report = SeedsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("exact", "no change", "asks", "guards")}
    for name, target, patches in _plants():
        _RUN.clear()
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
        finally:
            _RUN.clear()
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = len(report.cases) == 4 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 4 of 4 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str, ensure_ascii=False))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
