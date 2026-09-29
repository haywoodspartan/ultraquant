"""Choose distillation gaps from catalogue structure and promoted questions."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from ultraquant.interpreter.stash import _claim_provenance
from ultraquant.memory.factshards import normalize_subject


@dataclass(frozen=True)
class Target:
    category: str
    subject: str
    question: str
    attribute: str


def question_form(stash, attribute) -> tuple[str, str] | None:
    """Return the modal (category, question form) among promoted titles."""
    counts = Counter()
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        subject = fields.get("subject")
        title = entry.get("title") or ""
        if (fields.get("attribute") != attribute
                or not subject or subject not in title):
            continue
        provenance = _claim_provenance(entry)
        if provenance is None:
            continue
        category = provenance[1].split(":", 1)[0]
        before, after = title.split(subject, 1)
        form = _literal(before) + "{subject}" + _literal(after)
        counts[category, form] += 1
    return min(counts, key=lambda pair: (-counts[pair], pair)) if counts else None


def _literal(text: str) -> str:
    """Title text that str.format gives back unchanged."""
    return text.replace("{", "{{").replace("}", "}}")


def completion_targets(memory, stash, *, min_share=0.5,
                       min_support=5) -> list[Target]:
    """Fill missing attributes supported by directed subject co-occurrence."""
    holders = defaultdict(set)
    spelling = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        subject = record.get("subject")
        attribute = record.get("attribute")
        # The catalogue's own identity for a subject (§11.139), so "Iron"
        # and "iron" are one subject, asked about by its least spelling.
        same = normalize_subject(subject) if subject else ""
        if same and attribute:
            holders[attribute].add(same)
            spelling[same] = min(spelling.get(same, subject), subject)

    targets = {}
    forms = {}
    for source, source_subjects in holders.items():
        for attribute, subjects in holders.items():
            if source == attribute:
                continue
            support = len(source_subjects & subjects)
            if (support < min_support
                    or support / len(source_subjects) < min_share):
                continue
            if attribute not in forms:
                forms[attribute] = question_form(stash, attribute)
            if forms[attribute] is None:
                continue
            category, form = forms[attribute]
            for same in source_subjects - subjects:
                subject = spelling[same]
                targets[attribute, subject] = Target(
                    category, subject, form.format(subject=subject), attribute)
    return [targets[key] for key in sorted(targets)]
