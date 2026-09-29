"""Check held claims and stage revisions supported by two source lineages."""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re

from . import elicit, file, frontier, sources, targets
from ultraquant.memory.factshards import normalize_subject


def values_agree(a, b) -> bool:
    """Compare extracted numbers at their coarser precision, or normalized text."""
    a, b = elicit.extract(str(a)), elicit.extract(str(b))
    numbers = [re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
               for text in (a, b)]
    if all(numbers):
        left, right = (Decimal(match.group()) for match in numbers)
        quantum = Decimal(1).scaleb(max(left.as_tuple().exponent,
                                      right.as_tuple().exponent))
        with localcontext() as context:
            context.prec = max(len(left.as_tuple().digits),
                               len(right.as_tuple().digits)) + 1
            return (left.quantize(quantum, rounding=ROUND_HALF_UP)
                    == right.quantize(quantum, rounding=ROUND_HALF_UP))
    return elicit.agree(elicit.normalize(a), elicit.normalize(b))


def claims_to_check(memory, stash, source) -> list[dict]:
    """Select the newest eligible entry per key that still matches memory."""
    from . import corroborate

    checks = {}
    for entry in stash.entries(status="promoted"):
        teachers = (entry.get("provenance") or {}).get("teachers", [])
        if len(teachers) != 1 or teachers[0] == source:
            continue
        fields = entry.get("fields") or {}
        if any(fields.get(name) is None or str(fields[name]) == ""
               for name in ("key", "value", "subject", "attribute")):
            continue
        key = fields["key"]
        held = memory.recall_fact(key) or {}
        if held.get("value") is None or not corroborate.values_agree(
                held["value"], fields["value"]):
            continue
        if key not in checks or entry["id"] > checks[key]["entry_id"]:
            checks[key] = {"entry_id": entry["id"], "key": key,
                           "value": held["value"], "subject": fields["subject"],
                           "attribute": fields["attribute"]}
    return list(checks.values())


def _forward_target(stash, check):
    attribute, subject = check["attribute"], check["subject"]
    asking = targets.question_form(stash, attribute)
    category, form = (asking if asking is not None else (
        normalize_subject(attribute), frontier._seed_questions()["forward"]))
    return targets.Target(category, subject,
                          form.format(attribute=attribute, subject=subject), attribute)


def contest(memory, check) -> dict | None:
    """Queue both answers for a later source, leaving the held value alone."""
    return {name: check[name] for name in ("key", "value", "subject", "attribute")} | {
        "contest": check["answer"]}


def settled(ledger) -> set[str]:
    """Find contests followed by agreement or revision in stored ledger order."""
    contested, resolved = set(), set()
    for rows in ledger._read().values():
        for row in rows:
            check = row.get("check") or {}
            if (check.get("key") in contested
                    and check.get("verdict") in ("agreed", "revised")):
                resolved.add(check["key"])
            queued = row.get("queued") or {}
            if "contest" in queued:
                contested.add(queued["key"])
    return resolved


def contests(ledger, key) -> list[dict]:
    """Collect every unsettled contrary answer with its source, in ledger order."""
    from . import corroborate

    if key in corroborate.settled(ledger):
        return []
    return [{**queued, "source": source}
            for source, rows in ledger._read().items() for row in rows
            if (queued := row.get("queued")) is not None
            and "contest" in queued and queued["key"] == key]


def second_lineage(answer, contests) -> dict | None:
    """Return the first contrary answer that agrees with the new answer."""
    from . import corroborate

    return next((item for item in contests
                 if corroborate.values_agree(answer, item["contest"])), None)


def corroborate(memory, stash, teacher, ledger, source, *, records_path, run_id,
                confidence) -> dict:
    """Ask every unrecorded check together and persist its verdict."""
    # Resolve the module here because this function shares its name.
    from . import corroborate

    asked = ledger.asked(source)
    pairs = [(check, _forward_target(stash, check))
             for check in claims_to_check(memory, stash, source)]
    pairs = [(check, target) for check, target in pairs
             if elicit.question_id(target) not in asked]
    result = dict(checked=0, agreed=0, revised=0, contested=0, undecided=0, asked=0)
    if not pairs:
        return result
    items = [target for _, target in pairs]
    records = elicit.elicit(teacher, source, items, records_path)
    decisions = sources.single_source_decide(records, items)
    by_question = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)
    for check, target in pairs:
        qid = elicit.question_id(target)
        decision = decisions[qid]
        answer = (frontier._reverse_answer(by_question[qid], target, decision)[0]
                  if decision is not None else None)
        verdict = ("undecided" if decision is None else
                   "agreed" if corroborate.values_agree(answer, check["value"])
                   else "contested")
        check = {**check, "answer": answer}
        queued = None
        if verdict == "agreed":
            stash.add_teacher(check["entry_id"], source)
        elif verdict == "contested":
            # Re-asking a source under a changed category is not a new lineage.
            prior = [item for item in corroborate.contests(ledger, check["key"])
                     if item["source"] != source]
            found = corroborate.second_lineage(answer, prior)
            if found is not None:
                claim_form = (file.claim_form(stash, check["attribute"])
                              or file._seed()["claim"])
                slots = dict(subject=check["subject"], attribute=check["attribute"],
                             value=answer)
                stash.add_claim(
                    f"https://distill.invalid/{run_id}/{qid}", target.question,
                    claim_form.format(**slots), measured_confidence=confidence,
                    provenance={"run_id": run_id, "question_id": qid,
                                "teachers": [found["source"], source],
                                "lineages": [found["source"], source],
                                "settles": check["key"]},
                    fields={"key": check["key"], **slots})
                verdict = "revised"
            else:
                queued = corroborate.contest(memory, check)
        ledger.record(source, qid, verdict in ("agreed", "revised"), queued=queued, check={
            **{name: check[name] for name in
               ("key", "value", "subject", "attribute", "answer")},
            "verdict": verdict})
        result[verdict] += 1
    result.update(checked=len(pairs), asked=len(items))
    return result
