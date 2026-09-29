"""Read the teachers and unsettled answers behind a held fact."""

import json
from pathlib import Path

from . import corroborate, sources


def provenance(stash, ledger, key, value) -> dict:
    """Collect matching promoted teachers and distinct contests in ledger order."""
    teachers = {}
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        if (fields.get("key") == key
                and corroborate.values_agree(fields.get("value"), value)):
            recorded = entry.get("provenance") or {}
            names = recorded.get("teachers") or []
            teacher_ids = recorded.get("teacher_ids")
            if teacher_ids is None:
                teacher_ids = [sources.identity(name) for name in names]
            for name, teacher_id in zip(names, teacher_ids):
                teachers.setdefault(teacher_id, name)
    contests = []
    if ledger is not None:
        for item in corroborate.contests(ledger, key):
            if item["contest"] not in contests:
                contests.append(item["contest"])
    return {"teachers": sorted(teachers.values()), "contests": contests}


def note(found) -> str:
    """Format a provenance suffix using the stored wording."""
    words = json.loads((Path(__file__).with_name("data") / "provenance.json")
                       .read_text(encoding="utf-8"))
    if found["contests"]:
        return words["separator"] + words["contested"].format(
            answers=", ".join(found["contests"]))
    count = len(found["teachers"])
    if count >= 2:
        return words["separator"] + words["agree"].format(n=count)
    if count == 1:
        return words["separator"] + words["one"]
    return ""
