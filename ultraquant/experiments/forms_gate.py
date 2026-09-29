"""Claims filed in the library's own forms. The gate (indexing, stage 4).

``distill/file.py`` filed every distilled answer through four hand-typed
templates keyed by category, and the stash classified a claim as factual
through a regex listing the same four attributes. A completion target
(§11.144) for any other attribute raised KeyError, and would not have
been approved had it filed. Measured on the user's library:
- 384 promoted claims across four attributes;
- each attribute has exactly one claim form and one key form, with the
  subject and value slotted ("The atomic number of {subject} is
  {value}.", 75 of 75);
- 2 claims are unusable for learning, because a slot's text occurs twice
  ("The capital of Mexico is Mexico City.").

**The criteria, written before the run** - frozen in a pre-registration
(sha256 a2ca9d75...) before any code, with Amendment A (sha256
97b03233...), also before any code:
1. **Same filing as before**: the recorded runs (§11.130 and §11.134),
   replayed into a fresh stash and into one already holding the other
   run's promoted claims, file byte-identical entries to the committed
   template version, classification included.
2. **Forms from the library**: a stash whose promoted claims for an
   attribute use a distinctive form files a new target in that form and
   key.
3. **A new attribute files**: an attribute the stash has never seen files
   through the seed, as a factual claim, and after approval the catalogue
   answers it.
4. **No text in code**: file.py holds no claim, key or attribute text and
   no category name, and ``add_claim`` names no attribute. Claims without
   fields keep today's classification (Amendment A), which still reads
   the legacy regex.
5. **The exam can fail**:
   - P53 (the stash's forms ignored) breaches 2;
   - P54 (a category's attribute taken as its name) breaches 1;
   - P55 (no seed) breaches 3.
6. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**Reproduced first**: on the committed code, case 2 raised
KeyError('motto'), case 3 KeyError('currency'), and case 4 failed on the
templates.

**PASSED** on Claude's machine and in Astra's run: 4 of 4 cases, 3 of 3
plants caught. Both replays filed 384 entries identical to the committed
templates. The catalogue, auto-approve and distillation-replay gates
pass.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import Library, _scratch

__all__ = ["FormsReport", "run_gate"]


# -- the committed versions (437eedf), verbatim: the reference ----------------------

_STRUCTURED_CLAIM_COMMITTED = re.compile(
    r"^The (?:chemical symbol|capital|author|atomic number) of .+ is .+\.$",
    re.IGNORECASE,
)


def _add_claim_committed(self, url, title, claim, *, measured_confidence=None,
                         provenance=None, fields=None) -> int:
    from copy import deepcopy
    from urllib.parse import urlparse
    from ultraquant.interpreter.stash import _utc_now

    netloc = urlparse(url).netloc or url
    entry_id = self._next_id
    classification = ("factual-claim" if _STRUCTURED_CLAIM_COMMITTED.fullmatch(claim)
                      else self.classify(claim))
    self._entries[entry_id] = {
        "id": entry_id, "url": url, "netloc": netloc, "title": title,
        "fetched": _utc_now(), "claim": claim,
        "classification": classification, "status": "staged",
        "sources": [netloc], "notes": "",
        "measured_confidence": measured_confidence,
        "provenance": deepcopy(provenance),
        "fields": deepcopy(fields),
    }
    self._next_id += 1
    self.save()
    return entry_id


def _file_distilled_committed(stash, records, items, confidence, run_id) -> list[int]:
    from collections import Counter, defaultdict
    from ultraquant.distill.elicit import (agree, decide, extract, is_position,
                                           normalize, question_id)
    from ultraquant.interpreter.stash import _claim_provenance

    items = list(items)
    probes = [question_id(item) for item in items
              if getattr(item, "fictitious", False)]
    if probes:
        raise ValueError("Cannot file fictitious items: " + ", ".join(probes))
    records = list(records)
    decisions = decide(records, items)
    by_question = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)
    templates = {
        "symbol": "The chemical symbol of {subject} is {value}.",
        "capital": "The capital of {subject} is {value}.",
        "author": "The author of {subject} is {value}.",
        "number": "The atomic number of {subject} is {value}.",
    }
    filed = []
    for item in items:
        qid = question_id(item)
        if any(_claim_provenance(entry) == (run_id, qid) for entry in stash.entries()):
            continue
        promoted = decisions[qid]
        if promoted is None:
            continue
        samples = [r for r in by_question[qid]
                   if is_position(r.raw, category=item.category)
                   and agree(normalize(extract(r.raw)), promoted)]
        forms = Counter(extract(r.raw) for r in samples)
        value = min(forms, key=lambda form: (-forms[form], form))
        claim = templates[item.category].format(subject=item.subject, value=value)
        attribute = templates[item.category].split(" of {subject}")[0].lower()
        for article in ("the ", "a ", "an "):
            if attribute.startswith(article):
                attribute = attribute[len(article):]
                break
        key = templates[item.category].split(" is {value}")[0].format(
            subject=item.subject).lower()
        for article in ("the ", "a ", "an "):
            if key.startswith(article):
                key = key[len(article):]
                break
        filed.append(stash.add_claim(
            f"https://distill.invalid/{run_id}/{qid}", item.question, claim,
            measured_confidence=confidence,
            provenance={"teachers": sorted({r.teacher for r in samples}),
                        "lineages": sorted({r.lineage for r in samples}),
                        "samples_agreeing": len(samples), "run_id": run_id,
                        "question_id": qid},
            fields={"key": key, "value": value, "subject": item.subject,
                    "attribute": attribute},
        ))
    return filed


# -- worlds ------------------------------------------------------------------------------

@contextlib.contextmanager
def _library(prefix: str):
    lib = Library(_scratch(prefix))
    try:
        yield lib
    finally:
        shutil.rmtree(lib.root, ignore_errors=True)


def _comparable(stash) -> list:
    """Every entry, less the one field that is a clock reading."""
    return [{k: v for k, v in entry.items() if k != "fetched"}
            for entry in stash.entries()]


def _replay(filer, learned: bool) -> list:
    from ultraquant.distill import elicit as E
    from ultraquant.experiments import distill_facts_gate as D
    from ultraquant.experiments import distill_hard_gate as DH
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    with _library("uq_forms_replay_") as lib:
        filer(lib.stash, E.load_records(D.RECORDS), list(K.KNOWN), 0.959, "11130")
        if learned:                     # the second run meets promoted claims
            lib.approver().approve_all()
        filer(lib.stash, E.load_records(DH.RECORDS), list(H.KNOWN), 0.964, "11134")
        return _comparable(lib.stash)


def _agreeing(item, answer: str) -> list:
    """Three lineages, five samples each, all giving ``answer``."""
    from ultraquant.distill import elicit as E
    qid = E.question_id(item)
    return [E.Record(teacher, f"lineage-{n}", "exam.gguf", 1, qid, item.question,
                     seed, answer, E.normalize(E.extract(answer)))
            for n, teacher in enumerate(("t0", "t1", "t2")) for seed in E.SEEDS]


def _entry(stash, ids):
    wanted = set(ids or ())
    found = [entry for entry in stash.entries() if entry["id"] in wanted]
    return found[0] if len(found) == 1 else None


# -- the cases ---------------------------------------------------------------------------

_CACHE: dict = {}


def same_filing() -> bool:
    from ultraquant.distill import file as F
    from ultraquant.interpreter.stash import ContemporaryStash
    for learned in (False, True):
        if ("reference", learned) not in _CACHE:
            with mock.patch.object(ContemporaryStash, "add_claim", _add_claim_committed):
                _CACHE["reference", learned] = _replay(_file_distilled_committed, learned)
        if _replay(F.file_distilled, learned) != _CACHE["reference", learned]:
            return False
    return True


def forms_from_library() -> bool:
    from ultraquant.distill import file as F
    from ultraquant.distill.targets import Target
    with _library("uq_forms_learned_") as lib:
        for subject, value in (("Alpha", "Veritas"), ("Bravo", "Lux"),
                               ("Charlie", "Pax")):
            entry_id = lib.stash.add_claim(
                f"https://distill.invalid/exam/motto:{subject}",
                f"What is the motto of {subject}?",
                f"By decree, {subject} keeps the motto {value}.",
                measured_confidence=0.99,
                provenance={"run_id": "exam", "question_id": f"motto:{subject}"},
                fields={"key": f"decreed motto of {subject.lower()}", "value": value,
                        "subject": subject, "attribute": "motto"})
            # The world must not depend on the classification under test.
            lib.stash.promote(entry_id, lib.memory, force=True)
        target = Target("motto", "Delta", "What is the motto of Delta?", "motto")
        ids = F.file_distilled(lib.stash, _agreeing(target, "Fortis"), [target],
                               0.99, "exam-2")
        entry = _entry(lib.stash, ids)
        return (entry is not None
                and entry["claim"] == "By decree, Delta keeps the motto Fortis."
                and entry["fields"] == {"key": "decreed motto of delta",
                                        "value": "Fortis", "subject": "Delta",
                                        "attribute": "motto"})


def new_attribute_files() -> bool:
    from ultraquant.distill import file as F
    from ultraquant.distill.targets import Target
    with _library("uq_forms_new_") as lib:
        target = Target("currency", "Veltra", "What is the currency of Veltra?",
                        "currency")
        ids = F.file_distilled(lib.stash, _agreeing(target, "Lumen"), [target],
                               0.99, "exam-3")
        entry = _entry(lib.stash, ids)
        filed = (entry is not None
                 and entry["claim"] == "The currency of Veltra is Lumen."
                 and entry["fields"] == {"key": "currency of veltra", "value": "Lumen",
                                         "subject": "Veltra", "attribute": "currency"}
                 and entry["classification"] == "factual-claim")
        lib.approver().approve_all()
        lib.memory.save()
        lib.open()
        answer = lib.memory.catalogue_answer("What is the currency of Veltra?")
        return (filed and answer is not None and answer["form"] == "exact"
                and answer["record"]["value"] == "Lumen")


def _strings(tree) -> list:
    documentation = {id(node.body[0].value) for node in ast.walk(tree)
                     if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                          ast.AsyncFunctionDef))
                     and node.body and isinstance(node.body[0], ast.Expr)
                     and isinstance(getattr(node.body[0], "value", None), ast.Constant)}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in documentation]


def no_text_in_code() -> bool:
    from ultraquant.distill import file as F
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    from ultraquant.interpreter import stash as S
    categories = {item.category for item in list(K.KNOWN) + list(H.KNOWN)}
    attributes = {"chemical symbol", "capital", "author", "atomic number"}
    typed = [text for text in _strings(ast.parse(Path(F.__file__).read_text(encoding="utf-8")))
             if (" of " in text and (" is " in text or "{value}" in text))
             or text.lower() in categories | attributes]
    module = ast.parse(Path(S.__file__).read_text(encoding="utf-8"))
    add_claim = next(node for node in ast.walk(module)
                     if isinstance(node, ast.FunctionDef) and node.name == "add_claim")
    named = [text for text in _strings(add_claim)
             if any(attribute in text.lower() for attribute in attributes)]
    return not typed and not named


CASES = {
    "1 same filing as before": same_filing,
    "2 forms from the library": forms_from_library,
    "3 a new attribute files": new_attribute_files,
    "4 no text in code": no_text_in_code,
}


def _plants():
    from ultraquant.distill import file as F
    return [
        ("P53 the stash's forms ignored", "2 forms from the library",
         [mock.patch.object(F, "claim_form", lambda stash, attribute: None),
          mock.patch.object(F, "key_form", lambda stash, attribute: None)]),
        ("P54 a category's attribute taken as its name", "1 same filing as before",
         [mock.patch.object(F, "category_attribute", lambda stash, category: category)]),
        ("P55 no seed", "3 a new attribute files",
         [mock.patch.object(F, "_seed", lambda: {})]),
    ]


@dataclass
class FormsReport:
    """Whether distilled claims file in the library's own forms.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> FormsReport:
    report = FormsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
