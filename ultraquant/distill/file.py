"""File agreed distilled answers as measured, quarantined claims."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from ultraquant.distill.elicit import (
    agree, decide, extract, is_position, normalize, question_id,
)
from ultraquant.interpreter.stash import _claim_provenance


def _slot_form(text, slots) -> str | None:
    """Escape literal braces and replace unique, disjoint slot occurrences."""
    spans = []
    for name, value in slots.items():
        if value is None or value == "":
            return None
        value = str(value)
        start = text.find(value)
        if start < 0 or text.find(value, start + 1) >= 0:
            return None
        spans.append((start, start + len(value), name))
    spans.sort()
    parts = []
    end = 0
    for start, stop, name in spans:
        if start < end:
            return None
        parts.append(text[end:start].replace("{", "{{").replace("}", "}}"))
        parts.append("{" + name + "}")
        end = stop
    parts.append(text[end:].replace("{", "{{").replace("}", "}}"))
    return "".join(parts)


def _most_common(counts) -> str | None:
    return min(counts, key=lambda form: (-counts[form], form)) if counts else None


def claim_form(stash, attribute) -> str | None:
    """Return the modal claim form learned from promoted entries."""
    counts = Counter()
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        if fields.get("attribute") != attribute:
            continue
        form = _slot_form(entry.get("claim") or "", {
            "subject": fields.get("subject"), "value": fields.get("value"),
        })
        if form is not None:
            counts[form] += 1
    return _most_common(counts)


def key_form(stash, attribute) -> str | None:
    """Return the modal key form learned from promoted entries."""
    counts = Counter()
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        if fields.get("attribute") != attribute:
            continue
        form = _slot_form(fields.get("key") or "", {
            "subject": (fields.get("subject") or "").lower(),
        })
        if form is not None:
            counts[form] += 1
    return _most_common(counts)


def category_attribute(stash, category) -> str | None:
    """Return the modal attribute of a promoted provenance category."""
    counts = Counter()
    for entry in stash.entries(status="promoted"):
        attribute = (entry.get("fields") or {}).get("attribute")
        provenance = _claim_provenance(entry)
        if (attribute and provenance is not None
                and provenance[1].split(":", 1)[0] == category):
            counts[attribute] += 1
    return _most_common(counts)


def _seed() -> dict:
    """Read the fallback forms and category mapping from data."""
    path = Path(__file__).with_name("data") / "seed_forms.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _already_filed(stash, run_id, qid) -> bool:
    return any(_claim_provenance(entry) == (run_id, qid)
               for entry in stash.entries())


def _reject_probes(items) -> None:
    probes = [question_id(item) for item in items
              if getattr(item, "fictitious", False)]
    if probes:
        raise ValueError("Cannot file fictitious items: " + ", ".join(probes))


def file_distilled(stash, records, items, confidence, run_id,
                   min_lineages: int = 2) -> list[int]:
    items = list(items)
    # Review 6: reject the entire batch before any filing can take place.
    _reject_probes(items)
    records = list(records)
    # §11.155: one source at a time files with min_lineages=1, said in provenance.
    decisions = decide(records, items, min_lineages=min_lineages)
    by_question = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)
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
        attribute = (getattr(item, "attribute", None)
                     or category_attribute(stash, item.category)
                     or _seed().get("categories", {}).get(item.category))
        if not attribute:
            continue
        claim_template = claim_form(stash, attribute) or _seed().get("claim")
        key_template = key_form(stash, attribute) or _seed().get("key")
        if not claim_template or not key_template:
            continue
        claim = claim_template.format(attribute=attribute, subject=item.subject,
                                      value=value)
        key = key_template.format(attribute=attribute, subject=item.subject).lower()
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
