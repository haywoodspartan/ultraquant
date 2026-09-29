"""Recorded source progress and calibration against the library's held facts."""

from __future__ import annotations

import json
import math
import random
import tempfile
import urllib.request
from pathlib import Path

from . import elicit, targets
from .targets import Target
from .teachers import TeacherSpec


class LMStudioTeacher:
    """An already-loaded teacher; replay uses recorded replies, not seed promises."""

    def __init__(self, model: str, gguf: str | Path,
                 base_url: str = "http://127.0.0.1:1234/v1", timeout: float = 300,
                 options=None):
        self.spec = TeacherSpec(model, Path(gguf))
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = (dict(options) if options is not None else json.loads(
            (Path(__file__).with_name("data") / "sources.json").read_text(
                encoding="utf-8")).get(model, {}))

    def ask(self, questions, *, system, samples, temperature, top_p,
            max_tokens, seeds) -> list[list[str]]:
        seeds = tuple(seeds)
        if samples < 0 or len(seeds) != samples:
            raise ValueError("Each sample requires one seed")
        replies = []
        for question in questions:
            row = []
            for seed in seeds:
                payload = {
                    "model": self.spec.name,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": question}],
                    "temperature": temperature, "top_p": top_p,
                    "max_tokens": max_tokens, "seed": seed,
                }
                payload.update(self.options)
                request = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise ValueError("Completion must be an object")
                choices = result.get("choices")
                if (not isinstance(choices, list) or len(choices) != 1
                        or not isinstance(choices[0], dict)
                        or not isinstance(choices[0].get("message"), dict)
                        or not isinstance(choices[0]["message"].get("content"), str)):
                    raise ValueError("Completion must contain one raw message string")
                row.append(choices[0]["message"]["content"])
            replies.append(row)
        return replies


class SourceLedger:
    """Append ordered asks, replacing the JSON file only after a complete write."""

    def __init__(self, path):
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def record(self, source, question_id, promoted: bool, queued=None, check=None):
        data = self._read()
        row = {"question_id": question_id, "promoted": promoted}
        if queued is not None:
            row["queued"] = queued
        if check is not None:
            row["check"] = check
        data.setdefault(source, []).append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=self.path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def asked(self, source) -> set[str]:
        return {row["question_id"] for row in self.history(source)}

    def history(self, source) -> list[dict]:
        return self._read().get(source, [])


def used_up(ledger, source, frontier_ids, *, window=20, floor=0.2) -> bool:
    if window <= 0:
        raise ValueError("Window must be positive")
    if set(frontier_ids) <= ledger.asked(source):
        return True
    history = ledger.history(source)
    return (len(history) >= window
            and sum(row["promoted"] for row in history[-window:]) / window < floor)


def calibration_items(memory, stash, *, k=40, seed=155) -> list[tuple[Target, str]]:
    """Sample structured facts with question forms already held in the stash."""
    pairs = []
    forms = {}
    for key in sorted(memory.fact_keys()):
        record = memory.recall_fact(key) or {}
        subject, attribute = record.get("subject"), record.get("attribute")
        if not subject or not attribute or record.get("value") is None:
            continue
        if attribute not in forms:
            forms[attribute] = targets.question_form(stash, attribute)
        if forms[attribute] is None:
            continue
        category, form = forms[attribute]
        pairs.append((Target(category, subject, form.format(subject=subject), attribute),
                      str(record["value"])))
    pairs.sort(key=lambda pair: (pair[0].attribute, pair[0].subject))
    return random.Random(seed).sample(pairs, min(k, len(pairs)))


def calibrate(records, pairs) -> dict:
    pairs = list(pairs)
    decisions = elicit.decide(records, [target for target, _ in pairs], min_lineages=1)
    promoted = right = 0
    for target, value in pairs:
        answer = decisions[elicit.question_id(target)]
        if answer is not None:
            promoted += 1
            right += elicit.agree(elicit.normalize(answer), elicit.normalize(value))
    lower = 0.0
    if promoted:
        z = 1.959964
        p = right / promoted
        z2 = z * z
        lower = ((p + z2 / (2 * promoted)
                  - z * math.sqrt(p * (1 - p) / promoted + z2 / (4 * promoted**2)))
                 / (1 + z2 / promoted))
    return {"asked": len(pairs), "promoted": promoted, "right": right,
            "wilson_lower": max(0.0, lower)}


def single_source_decide(records, targets):
    """Use the ordinary filter with one lineage; callers access this via the module."""
    return elicit.decide(records, targets, min_lineages=1)
