"""The dictionary names a kind the teachers could not. The gate.

The frontier grows new properties per kind (§11.161): "Name one measurable
property that every {kind} has.", probed on five members. The kind comes
from the teachers ("What kind of thing is {a}?", a label three subjects
share) - and on the user's library ``capital`` (countries, US states,
provinces) never got one, so none of its 283 subjects ever grew a property.
Here, when the teachers name no kind, the dictionary does: the most specific
concept at least half of the subjects' senses reach.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 35324eef...) before the implementation existed. The world: a copy of
the pinned library after §11.155's filing, a lexicon, and a scripted first
source ("element" for an element's kind, the subject's own name otherwise;
"melting point" as the property of an element, "area" of any other kind;
"1234" for a property's value; "UNKNOWN" to anything else), studying until
used up.
1. **The kind the teachers could not name**: capital's learned kind is the
   exam's own ``dictionary_kind`` of its subjects; a property question named
   it; capital holds "area" adopted.
2. **The teachers' kind stands**: atomic number's kind is "element" and
   "melting point" is adopted.
3. **None stays none**: author stores None and has no property.
4. **No lexicon, no change**: without one, capital's kind is None, no
   property question names the dictionary's kind, and 2 holds as before.
5. **The exam can fail**: P132 (kind_of ignoring the lexicon) breaches 1;
   P133 (the most general ancestor instead of the most specific) breaches 1.
6. **Nothing regresses**: checked outside this module.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["DictKindReport", "run_gate"]

_KIND = re.compile(r"What kind of thing is (.+)\?")
_PROPERTY = re.compile(r"Name one measurable property that every (.+) has\.")
_VALUE = re.compile(r"What is the (melting point|area) of (.+)\?")
_RUN: dict = {}


def _seeds_ref():
    return importlib.import_module("ultraquant.experiments.seeds_gate")


def _expected_kind(lexicon, subjects) -> str | None:
    """The design's dictionary kind, read independently of the implementation."""
    S = _seeds_ref()
    depth = defaultdict(dict)
    for subject in subjects:
        for sense in S._noun_senses(lexicon, subject):
            level, seen, frontier = 0, {sense}, [sense]
            while frontier:
                level += 1
                following = []
                for sid in frontier:
                    record = lexicon.synset(sid) or {}
                    for parent in (*record.get("instance_hypernym", ()), *record.get("hypernym", ())):
                        if parent not in seen:
                            seen.add(parent)
                            following.append(parent)
                            depth[parent][subject] = min(depth[parent].get(subject, level), level)
                frontier = following
    reached = [k for k in depth if 2 * len(depth[k]) >= len(subjects)]
    if not reached:
        return None
    best = min(reached, key=lambda k: (sum(depth[k].values()) / len(depth[k]), -len(depth[k]), k))
    members = (lexicon.synset(best) or {}).get("members", [])
    return members[0] if members else None


class _Scripted:
    """The first source, scripted: kinds, properties and values; UNKNOWN otherwise."""

    def __init__(self, spec, elements):
        from ultraquant.lexicon import normalize_word
        self.spec, self.asked = spec, []
        self._norm = normalize_word
        self.elements = {normalize_word(e) for e in elements}

    def ask(self, questions, *, samples, **kwargs):
        rows = []
        for question in questions:
            self.asked.append(question)
            kind = _KIND.fullmatch(question)
            prop = _PROPERTY.fullmatch(question)
            value = _VALUE.fullmatch(question)
            if kind:
                answer = "element" if self._norm(kind[1]) in self.elements else kind[1]
            elif prop:
                answer = "melting point" if prop[1] == "element" else "area"
            elif value:
                answer = "1234"
            else:
                answer = "UNKNOWN"
            rows.append([answer] * samples)
        return rows


def _study(with_lexicon: bool) -> dict:
    key = f"study {with_lexicon}"
    if key in _RUN:
        return _RUN[key]
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    try:
        with _live_copy() as lib:
            lexicon = None
            if with_lexicon:
                importlib.import_module("ultraquant.lexicon.build").build(lib.root)
                lexicon = importlib.import_module("ultraquant.lexicon.lexicon").Lexicon.open(lib.root)
            subjects = defaultdict(set)
            for k in lib.memory.fact_keys():
                record = lib.memory.recall_fact(k) or {}
                if record.get("subject") and record.get("attribute"):
                    subjects[record["attribute"]].add(record["subject"])
            expected = (_expected_kind(lexicon, subjects["capital"]) if lexicon is not None else
                        None)
            teacher = _Scripted(GG._Teacher().spec, subjects["atomic number"])
            ledger = sources.SourceLedger(lib.root / "sources.json")
            rounds = []
            for n in range(1, 16):
                scratch = Path(tempfile.mkdtemp(prefix="uq_dictkind_"))
                kwargs = {"lexicon": lexicon} if lexicon is not None else {}
                result = frontier.study_round(
                    lib.memory, lib.stash, teacher, ledger, TG.FIRST,
                    confidence=_decided()[3]["wilson_lower"], run_id=f"dictkind-{n}",
                    records_path=scratch / f"{n}.jsonl", approver=lib.approver(), **kwargs)
                rounds.append({k: result[k] for k in ("asked", "proposed", "adopted", "refused", "used_up")})
                if result["used_up"]:
                    break
            vocabulary = {name: {k: item.get(k, "<absent>") for k in ("kind", "properties")}
                          for name, item in lib.memory._attribute_vocabulary().items()}
            property_questions = sorted({q for q in teacher.asked if _PROPERTY.fullmatch(q)})
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
    _RUN[key] = {"expected capital kind": expected, "vocabulary": vocabulary,
                 "property questions": property_questions, "rounds": rounds}
    return _RUN[key]


def _item(run, attribute):
    return run["vocabulary"].get(attribute, {})


# -- criteria -----------------------------------------------------------------------------

def named_by_dictionary() -> bool:
    run = _study(True)
    expected = run["expected capital kind"]
    capital = _item(run, "capital")
    asked = f"Name one measurable property that every {expected} has."
    return (expected is not None and capital.get("kind") == expected
            and asked in run["property questions"]
            and (capital.get("properties") or {}).get("area") == "adopted")


def teachers_stand() -> bool:
    run = _study(True)
    number = _item(run, "atomic number")
    return (number.get("kind") == "element"
            and (number.get("properties") or {}).get("melting point") == "adopted")


def none_stays_none() -> bool:
    run = _study(True)
    author = _item(run, "author")
    return author.get("kind") is None and author.get("properties") in (None, "<absent>", {})


def no_lexicon_no_change() -> bool:
    plain = _study(False)
    expected = _study(True)["expected capital kind"]
    capital = _item(plain, "capital")
    number = _item(plain, "atomic number")
    return (capital.get("kind") is None
            and f"Name one measurable property that every {expected} has." not in plain["property questions"]
            and number.get("kind") == "element"
            and (number.get("properties") or {}).get("melting point") == "adopted")


CASES = {"1 named by the dictionary": named_by_dictionary, "2 the teachers' kind stands": teachers_stand,
         "3 none stays none": none_stays_none, "4 no lexicon, no change": no_lexicon_no_change}


# -- plants --------------------------------------------------------------------------------

def _plants():
    from ultraquant.distill import frontier
    seeds = importlib.import_module("ultraquant.distill.seeds")
    real_kind_of = frontier.kind_of

    def deaf(memory, attribute, teacher, *, seed=158, lexicon=None):
        return real_kind_of(memory, attribute, teacher, seed=seed)

    def most_general(lexicon, subjects):
        S = _seeds_ref()
        depth = defaultdict(dict)
        for subject in subjects:
            for sense in S._noun_senses(lexicon, subject):
                for level, k in enumerate(S._ancestors(lexicon, sense), 1):
                    depth[k][subject] = min(depth[k].get(subject, level), level)
        reached = [k for k in depth if 2 * len(depth[k]) >= len(subjects)]
        if not reached:
            return None
        best = max(reached, key=lambda k: (sum(depth[k].values()) / len(depth[k]), k))
        return ((lexicon.synset(best) or {}).get("members") or [None])[0]

    return [
        ("P132 kind_of ignoring the lexicon", "1 named by the dictionary",
         [mock.patch.object(frontier, "kind_of", deaf)]),
        ("P133 the most general ancestor", "1 named by the dictionary",
         [mock.patch.object(seeds, "dictionary_kind", most_general)]),
    ]


@dataclass
class DictKindReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> DictKindReport:
    report = DictKindReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("study True", "study False")}
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
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = len(report.cases) == 4 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 2 of 2 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
