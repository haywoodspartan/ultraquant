"""§11.139: catalogue old distilled facts using their recorded provenance."""

from ultraquant.interpreter.stash import _claim_provenance, _split_claim


# §11.139: share the stash's split; structure is never guessed from grammar.
def _entry_pair(entry: dict) -> tuple[str, str] | None:
    fields = entry.get("fields") or {}
    if "key" in fields and "value" in fields:
        return fields["key"], fields["value"]
    split = _split_claim(entry["claim"], structured="provenance" in entry)
    if "key" in fields:
        return fields["key"], fields.get("value", split[1] if split else entry["claim"])
    return split


def _structure_of(entry) -> tuple[str, str] | None:
    """Subject from provenance, attribute from the exact trailing subject."""
    provenance = _claim_provenance(entry)
    if provenance is None:
        return None
    _category, separator, subject = provenance[1].partition(":")
    pair = _entry_pair(entry)
    if not separator or not subject or pair is None:
        return None
    key = pair[0]
    suffix = " of " + subject.lower()
    if not key.endswith(suffix):
        return None
    return subject, key[:-len(suffix)]


def structure_from_provenance(memory, stash) -> dict:
    """Fill missing catalogue slots and persist the stash before the memory."""
    counts = {"facts": 0, "entries": 0}
    for entry in stash.entries(status="promoted"):
        structure = _structure_of(entry)
        if structure is None:
            continue
        key, value = _entry_pair(entry)
        record = memory._fact_record(key)
        subject, attribute = structure
        # §11.139: complete provenance even if truth maintenance dropped the fact.
        if record is not None and (not record.get("subject") or not record.get("attribute")):
            record = dict(record)
            if not record.get("subject"):
                record["subject"] = subject
            if not record.get("attribute"):
                record["attribute"] = attribute
            memory.restore_fact(key, record)
            counts["facts"] += 1
        fields = dict(entry.get("fields") or {})
        fields.update(key=key, value=value, subject=subject, attribute=attribute)
        if fields != entry.get("fields"):
            stash._entries[entry["id"]]["fields"] = fields
            counts["entries"] += 1
    stash.save()
    memory.save()
    return counts


# §11.141: backfill asking evidence from promoted entries, without parsing.
def learn_question_forms(memory, stash) -> dict:
    """Learn eligible promoted titles; return counts of entries and slots read.

    Repeated runs count the same inputs but add no duplicate evidence. The
    caller persists the memory through its normal save/flush transaction.
    """
    from ultraquant.memory.factshards import normalize_subject

    count, subjects, attributes = 0, set(), set()
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        subject, attribute = fields.get("subject"), fields.get("attribute")
        title = entry.get("title") or ""
        if not subject or not attribute or not title.strip():
            continue
        memory.learn_asking(attribute, subject, title)
        count += 1
        subjects.add(normalize_subject(subject))
        attributes.add(normalize_subject(attribute))
    return {"entries": count, "subjects": len(subjects), "attributes": len(attributes)}
