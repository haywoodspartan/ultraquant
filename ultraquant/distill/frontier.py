"""Ask about missing members of dense ranges in the held catalogue."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from . import elicit, file, frontier, sources, targets
from ultraquant.interpreter import autoapprove
from ultraquant.interpreter.stash import _claim_provenance
from ultraquant.memory.factshards import normalize_subject


def _seed_questions() -> dict:
    """Question forms are data; absent forms cannot produce questions."""
    path = Path(__file__).with_name("data") / "seed_questions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def dense_window(values: list[int]) -> tuple[int, int] | None:
    """Choose observed endpoints by distinct support, width, then lower bound."""
    values = sorted(set(values))
    if len(values) < 5:
        return None
    best = None
    window = None
    for start, lo in enumerate(values):
        for end in range(start, len(values)):
            hi = values[end]
            count, width = end - start + 1, hi - lo + 1
            if 5 * count >= 4 * width:
                rank = (count, width, -lo)
                if best is None or rank > best:
                    best, window = rank, (lo, hi)
    return window


def sequence_gaps(memory) -> list[dict]:
    """Find integer gaps and retain outside values for later verification."""
    vocabulary = memory._attribute_vocabulary()
    rows = defaultdict(list)
    for key in sorted(memory.fact_keys()):
        record = memory.recall_fact(key) or {}
        attribute = normalize_subject(record.get("attribute") or "")
        if attribute in vocabulary and record.get("value") is not None:
            rows[attribute].append((key, record["value"]))
    gaps = []
    for attribute in sorted(rows):
        held = rows[attribute]
        integers = []
        for key, value in held:
            text = str(value).strip()
            if text.isdigit():
                try:
                    integers.append((key, value, int(text)))
                except ValueError:
                    # Some Unicode digits are not positional integer strings.
                    continue
        if 5 * len(integers) < 4 * len(held):
            continue
        window = dense_window([number for _, _, number in integers])
        if window is None:
            continue
        lo, hi = window
        numbers = {number for _, _, number in integers}
        gaps.append({
            "attribute": vocabulary[attribute].get("name", attribute),
            "window": window,
            "missing": [number for number in range(lo, hi + 1)
                        if number not in numbers],
            "outliers": [(key, value) for key, value, number in integers
                         if not lo <= number <= hi],
        })
    return gaps


@dataclass(frozen=True)
class ReverseTarget:
    category: str
    subject: str
    question: str
    attribute: str
    value: str


def kind_of(memory, attribute, teacher, *, seed=158) -> str | None:
    """Reuse a learned kind, or choose a label shared by three subjects."""
    normalized = normalize_subject(attribute)
    kind = memory._attribute_vocabulary().get(normalized, {}).get("kind")
    if kind:
        return kind
    form = _seed_questions().get("kind")
    if not form:
        return None
    subjects = set()
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        if (normalize_subject(record.get("attribute") or "") == normalized
                and record.get("subject")):
            subjects.add(record["subject"])
    if len(subjects) < 3:
        return None
    members = random.Random(seed).sample(sorted(subjects), 3)
    replies = teacher.ask(
        [form.format(a=member) for member in members], system=elicit.SYSTEM,
        samples=elicit.SAMPLES, temperature=elicit.TEMPERATURE,
        top_p=elicit.TOP_P, max_tokens=elicit.MAX_TOKENS, seeds=elicit.SEEDS)
    if len(replies) != 3 or any(len(row) != elicit.SAMPLES for row in replies):
        raise ValueError("Teacher returned an incomplete sample matrix")
    answers = [[answer for raw in row if elicit.is_position(raw)
                if (answer := elicit.normalize(elicit.extract(raw)))]
               for row in replies]
    counts = Counter(answer for row in answers for answer in row)
    shared = [candidate for candidate in counts
              if all(any(elicit.agree(answer, candidate) for answer in row)
                     for row in answers)]
    kind = min(shared, key=lambda candidate: (
        -sum(count for answer, count in counts.items()
             if elicit.agree(answer, candidate)),
        -counts[candidate], candidate), default=None)
    if kind is not None:
        memory.learn_kind(attribute, kind)
    return kind


def reverse_targets(memory, stash, teacher) -> list[ReverseTarget]:
    """Turn missing integers into questions using the learned kind label."""
    form = _seed_questions().get("reverse")
    if not form:
        return []
    result = []
    for gap in sequence_gaps(memory):
        attribute = gap["attribute"]
        asking = targets.question_form(stash, attribute)
        if not gap["missing"] or asking is None:
            continue
        kind = kind_of(memory, attribute, teacher)
        if kind is None:
            continue
        category, _ = asking
        for number in gap["missing"]:
            value = str(number)
            result.append(ReverseTarget(
                category, value, form.format(kind=kind, attribute=attribute,
                                             value=value), attribute, value))
    return result


def forward_value(memory, subject, attribute) -> str | None:
    """Read the held value using catalogue identities, not a guessed key."""
    return memory.held_value(subject, attribute)


def ask_forward(stash, teacher, source, pairs, records_path) -> dict[tuple[str, str], str | None]:
    """Ask distinct pairs together using the library's learned question forms."""
    items = []
    for subject, attribute in dict.fromkeys(pairs):
        asking = targets.question_form(stash, attribute)
        if asking is None:
            continue
        category, form = asking
        items.append(targets.Target(category, subject, form.format(subject=subject), attribute))
    if not items:
        return {}
    records = elicit.elicit(teacher, source, items, records_path)
    decisions = sources.single_source_decide(records, items)
    return {(item.subject, item.attribute): decisions[elicit.question_id(item)]
            for item in items}


def comes_back(forward, value) -> bool:
    """Require the forward answer to equal the requested value."""
    return (forward is not None
            and elicit.normalize(str(forward)) == elicit.normalize(str(value)))


def _reverse_answer(records, target, decision):
    """Choose the same extracted subject and supporting samples for both paths."""
    samples = [record for record in records
               if elicit.is_position(record.raw, category=target.category)
               and elicit.agree(elicit.normalize(elicit.extract(record.raw)), decision)]
    forms = Counter(elicit.extract(record.raw) for record in samples)
    subject = min(forms, key=lambda form: (-forms[form], form)) if forms else None
    return subject, samples


def round_trip(memory, stash, teacher, source, targets, records, records_path) -> dict[str, dict]:
    """Judge all decided reverse answers against the index or one forward batch."""
    records, targets = list(records), list(targets)
    decisions = sources.single_source_decide(records, targets)
    by_question = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)
    result, pairs = {}, []
    for target in targets:
        qid = elicit.question_id(target)
        decision = decisions[qid]
        if decision is None:
            continue
        subject, _ = _reverse_answer(by_question[qid], target, decision)
        forward = frontier.forward_value(memory, subject, target.attribute)
        origin = "index" if forward is not None else "source"
        result[qid] = {"subject": subject, "forward": forward, "forward_from": origin}
        if origin == "source":
            pairs.append((subject, target.attribute))
    answers = (frontier.ask_forward(stash, teacher, source, pairs, records_path)
               if pairs else {})
    for target in targets:
        row = result.get(elicit.question_id(target))
        if row is None:
            continue
        if row["forward_from"] == "source":
            row["forward"] = answers.get((row["subject"], target.attribute))
        row["back"] = frontier.comes_back(row["forward"], target.value)
    return result


def file_reverse(stash, records, targets, confidence, run_id) -> list[int]:
    """File the answer as the subject and the requested integer as its value."""
    records, targets = list(records), list(targets)
    decisions = sources.single_source_decide(records, targets)
    by_question = defaultdict(list)
    for record in records:
        by_question[record.question_id].append(record)
    filed = []
    for target in targets:
        qid = elicit.question_id(target)
        answer = decisions[qid]
        if answer is None or file._already_filed(stash, run_id, qid):
            continue
        subject, samples = _reverse_answer(by_question[qid], target, answer)
        if subject is None:
            continue
        claim_form = (file.claim_form(stash, target.attribute)
                      or file._seed().get("claim"))
        key_form = (file.key_form(stash, target.attribute)
                    or file._seed().get("key"))
        if not claim_form or not key_form:
            continue
        slots = dict(subject=subject, attribute=target.attribute, value=target.value)
        key = key_form.format(**slots).lower()
        filed.append(stash.add_claim(
            f"https://distill.invalid/{run_id}/{qid}", target.question,
            claim_form.format(**slots), measured_confidence=confidence,
            provenance={"run_id": run_id, "question_id": qid,
                        "teachers": sorted({r.teacher for r in samples}),
                        "lineages": sorted({r.lineage for r in samples}),
                        "samples_agreeing": len(samples)},
            fields={"key": key, **slots}))
    return filed


def verification_queue(memory, ledger=None) -> list[tuple[str, str]]:
    """Return outliers for a later source without revising held facts."""
    queued = [(key, str(value)) for gap in sequence_gaps(memory)
              for key, value in gap["outliers"]]
    if ledger is not None:
        for rows in ledger._read().values():
            for row in rows:
                claim = row.get("queued")
                if claim is not None:
                    queued.append((claim["key"], str(claim["value"])))
    return list(dict.fromkeys(queued))


def pending(memory, stash, teacher, ledger, source) -> list:
    """Return questions this source has not answered from the current frontier."""
    items = (targets.completion_targets(memory, stash)
             + frontier.reverse_targets(memory, stash, teacher))
    asked = ledger.asked(source)
    return [target for target in items if elicit.question_id(target) not in asked]


def study_round(memory, stash, teacher, ledger, source, *, confidence, run_id,
                records_path, approver=None) -> dict:
    """Ask the unvisited frontier, file its answers, and record promotions."""
    items = frontier.pending(memory, stash, teacher, ledger, source)
    forward = [t for t in items if not isinstance(t, ReverseTarget)]
    reverse = [t for t in items if isinstance(t, ReverseTarget)]
    records = (elicit.elicit(teacher, source, items, records_path)
               if items else [])
    trips = frontier.round_trip(memory, stash, teacher, source, reverse, records, records_path)
    filed = file.file_distilled(
        stash, records, forward, confidence, run_id, min_lineages=1)
    filed += frontier.file_reverse(
        stash, records, [t for t in reverse if trips.get(elicit.question_id(t), {}).get("back")],
        confidence, run_id)
    if approver is None:
        approver = autoapprove.AutoApprover(
            stash, memory, Path(records_path).with_suffix(".approvals.jsonl"))
    approver.approve_all()
    promoted = {_claim_provenance(entry)
                for entry in stash.entries(status="promoted")}
    queued = 0
    for target in items:
        qid = elicit.question_id(target)
        trip = trips.get(qid)
        if trip is not None and not trip["back"]:
            key_form = (file.key_form(stash, target.attribute)
                        or file._seed().get("key"))
            slots = {"subject": trip["subject"], "attribute": target.attribute,
                     "value": target.value}
            ledger.record(source, qid, False, queued={
                "key": key_form.format(**slots).lower(), **slots,
                "forward": trip["forward"], "forward_from": trip["forward_from"]})
            queued += 1
        else:
            ledger.record(source, qid, (run_id, qid) in promoted)
    regenerated = (targets.completion_targets(memory, stash)
                   + frontier.reverse_targets(memory, stash, teacher))
    return {"asked": len(items), "filed": len(filed), "queued": queued,
            "used_up": bool(sources.used_up(
                ledger, source, [elicit.question_id(t) for t in regenerated]))}
