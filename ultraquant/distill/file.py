"""File agreed distilled answers as measured, quarantined claims."""

from collections import Counter, defaultdict

from ultraquant.distill.elicit import (
    agree, decide, extract, is_position, normalize, question_id,
)
from ultraquant.interpreter.stash import _claim_provenance


def _already_filed(stash, run_id, qid) -> bool:
    return any(_claim_provenance(entry) == (run_id, qid)
               for entry in stash.entries())


def _reject_probes(items) -> None:
    probes = [question_id(item) for item in items
              if getattr(item, "fictitious", False)]
    if probes:
        raise ValueError("Cannot file fictitious items: " + ", ".join(probes))


def file_distilled(stash, records, items, confidence, run_id) -> list[int]:
    items = list(items)
    # Review 6: reject the entire batch before any filing can take place.
    _reject_probes(items)
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
        # §11.139: the template already names its attribute; keep that slot.
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
            # §11.139: retain the item's structure alongside the claim.
            fields={"key": key, "value": value, "subject": item.subject,
                    "attribute": attribute},
        ))
    return filed
