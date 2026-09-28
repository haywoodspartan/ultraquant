"""File agreed distilled answers as measured, quarantined claims."""

from collections import Counter, defaultdict

from ultraquant.distill.elicit import (
    agree, decide, extract, is_position, normalize, question_id,
)
from ultraquant.interpreter.stash import _claim_provenance


def _already_filed(stash, run_id, qid) -> bool:
    return any(_claim_provenance(entry) == (run_id, qid)
               for entry in stash.entries())


def file_distilled(stash, records, items, confidence, run_id) -> list[int]:
    records, items = list(records), list(items)
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
        if _already_filed(stash, run_id, qid):
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
            fields={"key": key, "value": value},
        ))
    return filed
