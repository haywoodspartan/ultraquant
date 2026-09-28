"""Answer-only elicitation and the pre-registered, answer-blind filter."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM = (
    "Answer with only the answer - a name, a word or a number - and nothing else. "
    "If you do not know, or if what is asked about does not exist, answer UNKNOWN."
)
SAMPLES = 5
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 24
SEEDS = (1, 2, 3, 4, 5)

_THINK = re.compile(r"<think\b[^>]*>.*?(?:</think\s*>|\Z)", re.I | re.S)
_REFUSAL = re.compile(
    r"\b(?:unknown|do not know|don t know|dont know|not sure|unsure|"
    r"not known|does not exist|doesn t exist|doesnt exist|not exist|"
    r"no such|there is no|there are no|fictional|fictitious|imaginary|"
    r"made up|invented|hypothetical|not a real|not real|not an actual|"
    r"no known|no record|no information|not aware|not familiar|"
    r"cannot answer|can t answer|cannot determine|unable to|"
    r"not recognized|not a recognized|not a known|no country|"
    r"no element|no novel|cannot provide|can t provide|"
    r"n a|not applicable|none known|no answer|not available|"
    r"no capital|no author|no symbol|no atomic number)\b"
)
_HEDGED_ANSWER = re.compile(r"\b(?:not|no|never|nor|or)\b")
CONFLICT = object()


def question_id(item) -> str:
    return f"{item.category}:{item.subject}"


def extract(raw: str) -> str:
    """Remove reasoning and unwrap the first nonempty answer line."""
    text = _THINK.sub("", raw)
    # Some servers expose serialized chat delimiters in message.content.
    # These are transport artifacts; Record.raw still preserves them exactly.
    text = re.sub(r"<\|[^|>\r\n]{1,64}\|>", "", text)
    text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    # Repeat to handle nested forms such as **Answer:** "Paris.".
    while True:
        previous = text
        text = text.strip().strip("*_`\"'\u2018\u2019\u201c\u201d").strip()
        text = re.sub(r"^answer\s*:\s*", "", text, flags=re.I)
        text = text.removesuffix(".").rstrip()
        if text == previous:
            return text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(char if char.isalnum() else " " for char in text)
    words = text.split()
    while words and words[0] in {"the", "a", "an"}:
        words.pop(0)
    return " ".join(words)


def is_abstention(text: str) -> bool:
    answer = normalize(extract(text))
    # Search the entire reply (outside reasoning), including later lines.
    return not answer or answer == "none" or bool(
        _REFUSAL.search(normalize(_THINK.sub("", text))))


def is_position(text: str, category=None) -> bool:
    if is_abstention(text):
        return False
    reply = _THINK.sub("", text)
    reply = re.sub(r"<\|[^|>\r\n]{1,64}\|>", "", reply)
    if category == "symbol" and re.fullmatch(r"[A-Za-z]{1,2}", extract(text)):
        # The answer line may be the symbol No. Later lines still veto it.
        lines = [line for line in reply.splitlines() if line.strip()]
        reply = "\n".join(lines[1:])
    return not _HEDGED_ANSWER.search(normalize(reply))


def held(answers, min_count: int = 3, category=None) -> str | None:
    """The modal answer, judged on whole replies.

    Each reply is tested for refusal as a whole, so "Veltra" followed by
    a line calling the country fictional is an abstention, not a
    position. Claude's review found the first version handed this only
    the first line.
    """
    counts = Counter(normalize(extract(answer)) for answer in answers
                     if is_position(answer, category=category)
                     and normalize(extract(answer)))
    if not counts:
        return None
    answer, count = counts.most_common(1)[0]
    return answer if count >= min_count else None


def agree(a: str, b: str) -> bool:
    # held() has already checked the whole reply and its category. Identical
    # symbols (including No) must survive the family and lineage reductions.
    if a == b and re.fullmatch(r"[a-z]{1,2}", a):
        return True
    if any(_HEDGED_ANSWER.search(normalize(answer)) for answer in (a, b)):
        return False
    if a == b:
        return True
    shorter, longer = sorted((a.split(), b.split()), key=len)
    return bool(shorter) and longer[-len(shorter):] == shorter


def family_position(helds: dict):
    """Keep a family's dissent distinct from its unanimous abstention."""
    answers = [answer for answer in helds.values() if answer is not None]
    if not answers:
        return None
    longest = max(answers, key=lambda answer: (len(answer.split()), len(answer)))
    return longest if all(agree(longest, answer) for answer in answers) else CONFLICT


def promote(held_by_lineage: dict, min_lineages: int = 2) -> str | None:
    if any(answer is CONFLICT for answer in held_by_lineage.values()):
        return None
    answers = [answer for answer in held_by_lineage.values() if answer is not None]
    if len(answers) < min_lineages or not answers:
        return None
    longest = max(answers, key=lambda answer: (len(answer.split()), len(answer)))
    return longest if all(agree(longest, answer) for answer in answers) else None


@dataclass
class Record:
    teacher: str
    lineage: str
    gguf_name: str
    gguf_size: int
    question_id: str
    question: str
    seed: int
    raw: str
    answer: str


def elicit(teacher, lineage, items, path, *,
           manifest: list[dict] | None = None) -> list[Record]:
    """Save teacher metadata, then append and flush every returned sample."""
    from . import teachers

    items = list(items)
    gguf = Path(teacher.spec.gguf)
    size = gguf.stat().st_size
    path = Path(path)
    if manifest is None:
        manifest = (teachers.load_manifest(path)
                    if path.exists() and path.with_suffix(".manifest.json").exists()
                    else [])
        entry = teachers._manifest_entry(teacher.spec.name, gguf,
                                         {teacher.spec.name: lineage})
        manifest = [row for row in manifest if row["name"] != teacher.spec.name]
        manifest.append(entry)
    teachers._save_manifest(path, manifest)
    replies = teacher.ask(
        [item.question for item in items], system=SYSTEM, samples=SAMPLES,
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seeds=SEEDS,
    )
    if len(replies) != len(items) or any(len(row) != SAMPLES for row in replies):
        raise ValueError("Teacher returned an incomplete sample matrix")
    if any(not isinstance(raw, str) for row in replies for raw in row):
        raise TypeError("Teacher samples must be raw strings")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with path.open("a", encoding="utf-8") as handle:
        for item, row in zip(items, replies):
            for seed, raw in zip(SEEDS, row):
                record = Record(teacher.spec.name, lineage, gguf.name, size,
                                question_id(item), item.question, seed, raw,
                                normalize(extract(raw)))
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
    return records


def load_records(path) -> list[Record]:
    with Path(path).open(encoding="utf-8") as handle:
        return [Record(**json.loads(line)) for line in handle if line.strip()]


def decide(records, items) -> dict[str, str | None]:
    """Recompute from raw samples, without consulting item answers."""
    by_question = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_question[record.question_id][record.teacher].append(record)
    decisions = {}
    for item in items:
        key = question_id(item)
        lineages = defaultdict(dict)
        for teacher, samples in by_question.get(key, {}).items():
            lineage = samples[0].lineage
            if any(record.lineage != lineage for record in samples):
                raise ValueError(f"Inconsistent lineage for teacher {teacher!r}")
            lineages[lineage][teacher] = held([r.raw for r in samples],
                                              category=item.category)
        # Multiple teachers contribute one family position, including dissent.
        positions = {lineage: family_position(answers)
                     for lineage, answers in lineages.items()}
        decisions[key] = promote(positions)
    return decisions
