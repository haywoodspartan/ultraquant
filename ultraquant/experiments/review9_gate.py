"""The ninth review, answered. The gate.

GPT-6 Astra's ninth adversarial review (of §11.141-§11.143) found three
faults. Each is reproduced here, on the committed code, before any fix:
1. [high] **A chat statement was answered as a question.** "I wrote
   Dune." is chat, but every word was "explained" ("I" a stopword,
   "wrote" a learned way of asking for author), so chat asserted
   "author of dune is Frank Herbert (confidence 0.96)." "I know the
   capital of Kenya." likewise. §11.142 took coverage as evidence of a
   request.
2. [high] **Missing knowledge resolved an ambiguity.** With "wrote"
   learned for both author and preface author, Dune holding only a
   preface author answered "Who wrote Dune?" exactly with it.
3. [medium] **A bounded lookup was taken as proof of absence.** The
   unknown-subject refusal trusted a top-10 search: ten better-ranked
   "freedonia dossier" keys, or an accented "café capital" (indexed as
   "caf"), hid a held fact behind "I don't hold that".

**The criteria, written before the run** - frozen in a pre-registration
(sha256 ee8dfc16...) before any fix, with Amendment A (sha256
d75c3f91...), also before any fix, and Amendment C (sha256 e123848c...),
recorded after the reproduction run and before any fix:
1. **Chat stays chat**: "I wrote Dune." and "I know the capital of
   Kenya." get no assertion answering them, "Tell me the capital of
   Australia." mentions Canberra and not Adelaide, and "Tell me the
   capital of Cote d'Ivoire." mentions Yamoussoukro, held under the
   accented "Côte d'Ivoire", without asserting it. Amendment C: a world
   check requires that chat's own choice (Recall's ladder) would NOT
   find Yamoussoukro, or the verdict is VOID. The reproduction run showed
   the first world check (Adelaide) could never hold: Recall offers the
   longest stored key in the message, the same fact the catalogue
   matches.
2. **No ambiguity resolved by absence**: "Who wrote Dune?" is not
   answered exactly with the preface author, in a session and on an
   in-RAM memory.
3. **No held fact hidden**: "What is the capital of Freedonia?" answers
   Fredville beside ten better-ranked dossier keys, "What is the capital
   of Café?" answers the chat-taught "café capital", and "What is the
   capital of the US state of Kessaway?" is still refused plainly, in a
   session and on an in-RAM memory.
4. **Nothing regresses**: §11.142's gate passes, with 0 wrong, at most
   68 hints and at most 18 unknown subjects named with another's fact
   (§11.141's recorded measures). The native parity gates are checked
   outside this module.
5. **The exam can fail**: P50 (the committed chat, asserting catalogue
   matches) breaches 1, P51 (the committed answer, judging ambiguity
   among held facts only) breaches 2, and P52 (the committed top-10
   refusal) breaches 3.

**Reproduced first**: on the committed code (437eedf), cases 1-3 failed
exactly as the review said, in a session and on an in-RAM memory.

**PASSED** on Claude's machine and in Astra's run: 4 of 4 cases, the
world check live, 3 of 3 plants caught. §11.142's gate still passes
("Tell me the capital of X." 100%, now as a mention; 0 wrong; 0 unknown
subjects named; 68 hints, all on the ungated form, as §11.141 recorded),
and the native parity gates show zero differences.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.answers_gate import _REFUSALS

__all__ = ["Review9Report", "run_gate"]


# -- the committed functions (437eedf), verbatim: the plants --------------------------

def _chat_437eedf(self, ctx) -> None:
    from ultraquant.interpreter.thoughts import _shown_value
    # §11.142: exact catalogue requests share the chat intent's reply path.
    answer = ctx.session.memory.catalogue_request(ctx.text)
    if answer is not None:
        key, record = answer["key"], answer["record"]
        ctx.say(f"{key} is {_shown_value(record)} "
                f"(confidence {record['confidence']:.2f}).")
        ctx.note(self.name, f"catalogue {answer['form']} answer {key!r}")
        return
    routes = ctx.data.get("routes", [])
    facts = ctx.data.get("facts", [])
    if facts:
        key, fact = facts[0]
        ctx.say(f"That lands near '{key}', which I hold as: "
                f"{_shown_value(fact)}.")
    elif routes:
        ctx.say(f"That reads as {routes[0][0]}. I have nothing stored on it yet.")
    else:
        ctx.say("I have nothing on that yet. ':help' lists what I can do.")
    ctx.note(self.name, "conversational reply")


def _unheld_subject_437eedf(self, words: set, known: set) -> bool:
    from ultraquant.memory.factshards import normalize_subject
    others = words - known
    if not others:
        return False
    for key in self.find_facts(" ".join(sorted(others)), top_k=10):
        record = self.recall_fact(key)
        if (record is not None and not record.get("subject")
                and others <= set(normalize_subject(key).split())):
            return False
    return True


def _catalogue_answer_437eedf(self, text: str) -> dict | None:
    from ultraquant.memory.factshards import (attribute_words, choose_subject,
                                              normalize_subject, question_words)
    subjects = self.subjects_in(text)
    chosen = (self.shards._choose_subject(subjects) if self.shards is not None
              else choose_subject(subjects))
    vocabulary = self._attribute_vocabulary()
    words = question_words(text, chosen or "")
    if chosen is not None:
        if self.shards is not None:
            keys = self.shards._index_data("subjects", chosen)[chosen]["keys"]
        else:
            keys = [key for key, record in self._facts.items()
                    if normalize_subject(record.get("subject") or "") == chosen]
        best, winners = 0, []
        for key in keys:
            record = self.recall_fact(key)
            if not record or not record.get("attribute"):
                continue
            attribute = normalize_subject(record["attribute"])
            explained = attribute_words(attribute, vocabulary.get(attribute, {}))
            score = len(words & explained)
            if score > best:
                best, winners = score, []
            if score == best and score >= 1:
                winners.append({"form": "exact" if words <= explained else "reading",
                                "key": key, "record": record})
        return winners[0] if len(winners) == 1 else None
    if subjects:
        return None
    known = set().union(*(attribute_words(attribute, item)
                          for attribute, item in vocabulary.items()))
    if words & known and self._unheld_subject(words, known):
        return {"form": "unknown-subject"}
    return None


# -- the worlds ------------------------------------------------------------------------

@contextlib.contextmanager
def _scratch(prefix: str):
    root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _session(root: Path):
    from ultraquant.interpreter.thoughts import build_session
    return build_session(root, seed=0)


def _ram(root: Path):
    from ultraquant.memory.systematic import SystematicMemory
    return SystematicMemory(path=root / "memory.json")


def _ask(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    reply, _trace = run_pipeline(text, session)
    return reply


def _chat_world(memory) -> None:
    memory.remember_fact("author of dune", "Frank Herbert", 0.96,
                         subject="Dune", attribute="author")
    memory.remember_fact("author of emma", "Jane Austen", 0.9,
                         subject="Emma", attribute="author")
    for subject in ("Dune", "Emma"):
        memory.learn_asking("author", subject, f"Who wrote {subject}?")
    memory.remember_fact("capital of kenya", "Nairobi", 0.9,
                         subject="Kenya", attribute="capital")
    memory.remember_fact("capital of australia", "Canberra", 0.6,
                         subject="Australia", attribute="capital")
    # The pre-registered trap, which the reproduction run showed Recall's
    # ladder never falls into (Amendment C).
    for _ in range(4):
        memory.remember_fact("capital of south australia", "Adelaide", 0.95,
                             subject="South Australia", attribute="capital")
    # Amendment C: a subject only the catalogue's folding can find.
    memory.remember_fact("capital of côte d'ivoire", "Yamoussoukro", 0.9,
                         subject="Côte d'Ivoire", attribute="capital")


def _preface_world(memory) -> None:
    """Astra's reproduction: Echo and Foxtrot hold both; Dune one."""
    for subject, author, preface in (("Echo", "Ann Author", "Pat Preface"),
                                     ("Foxtrot", "Ben Author", "Quinn Preface")):
        memory.remember_fact(f"author of {subject.lower()}", author, 0.9,
                             subject=subject, attribute="author")
        memory.remember_fact(f"preface author of {subject.lower()}", preface, 0.9,
                             subject=subject, attribute="preface author")
        memory.learn_asking("author", subject, f"Who wrote {subject}?")
        memory.learn_asking("preface author", subject,
                            f"Who wrote the preface of {subject}?")
    memory.remember_fact("preface author of dune", "Guest Writer", 0.8,
                         subject="Dune", attribute="preface author")


def _absence_world(memory, teach) -> None:
    """Astra's reproductions: a truncated search, and an accented key."""
    memory.remember_fact("capital of kenya", "Nairobi", 0.9,
                         subject="Kenya", attribute="capital")
    memory.remember_fact("capital of ghana", "Accra", 0.9,
                         subject="Ghana", attribute="capital")
    teach("freedonia capital", "Fredville")
    teach("café capital", "Cafeville")
    for n in range(1, 11):
        memory.remember_fact(f"freedonia dossier {n}", f"file {n}", 0.9,
                             subject=f"Holder {chr(64 + n)}", attribute="dossier")


def _teach_by_chat(session):
    def teach(key, value):
        _ask(session, f"{key} is {value}")
    return teach


def _teach_directly(memory):
    def teach(key, value):
        memory.remember_fact(key, value, 0.6)
    return teach


def _asserted(reply: str) -> bool:
    """An answer asserted as a fact, in either reply path's own words."""
    return "(confidence" in reply and not reply.startswith("Reading that as")


# -- the cases -------------------------------------------------------------------------

def chat_stays_chat() -> bool:
    with _scratch("uq_review9_chat_") as root:
        session = _session(root)
        _chat_world(session.memory)
        wrote = _ask(session, "I wrote Dune.")
        know = _ask(session, "I know the capital of Kenya.")
        australia = _ask(session, "Tell me the capital of Australia.")
        ivoire = _ask(session, _IVOIRE)
        return (not _asserted(wrote) and "(confidence" not in wrote
                and not _asserted(know) and "(confidence" not in know
                and "Canberra" in australia and "Adelaide" not in australia
                and "Yamoussoukro" in ivoire and "(confidence" not in ivoire)


_IVOIRE = "Tell me the capital of Cote d'Ivoire."


def world_is_live() -> bool:
    """Amendment C: chat's own choice cannot find what the catalogue can."""
    from ultraquant.memory import systematic as SM
    with _scratch("uq_review9_world_") as root, mock.patch.object(
            SM.SystematicMemory, "catalogue_request", lambda self, text: None):
        session = _session(root)
        _chat_world(session.memory)
        return "Yamoussoukro" not in _ask(session, _IVOIRE)


def no_ambiguity_by_absence() -> bool:
    with _scratch("uq_review9_preface_") as root:
        session = _session(root)
        _preface_world(session.memory)
        reply = _ask(session, "Who wrote Dune?")
        in_session = not (_asserted(reply) and "Guest Writer" in reply)
    with _scratch("uq_review9_preface_ram_") as root:
        memory = _ram(root)
        _preface_world(memory)
        answer = memory.catalogue_answer("Who wrote Dune?")
        in_ram = not (answer is not None and answer.get("form") == "exact"
                      and answer["record"]["value"] == "Guest Writer")
    return in_session and in_ram


def _refused_plainly(reply: str) -> bool:
    return reply.startswith(_REFUSALS) and "Nearest I hold" not in reply


def no_held_fact_hidden() -> bool:
    kessaway = "What is the capital of the US state of Kessaway?"
    with _scratch("uq_review9_absence_") as root:
        session = _session(root)
        _absence_world(session.memory, _teach_by_chat(session))
        in_session = ("Fredville" in _ask(session, "What is the capital of Freedonia?")
                      and "Cafeville" in _ask(session, "What is the capital of Café?")
                      and _refused_plainly(_ask(session, kessaway)))
    with _scratch("uq_review9_absence_ram_") as root:
        memory = _ram(root)
        _absence_world(memory, _teach_directly(memory))
        unknown = {"form": "unknown-subject"}
        in_ram = (memory.catalogue_answer("What is the capital of Freedonia?") != unknown
                  and memory.catalogue_answer("What is the capital of Café?") != unknown
                  and memory.catalogue_answer(kessaway) == unknown)
    return in_session and in_ram


_CACHE: dict = {}


def _requests() -> dict:
    if "r" not in _CACHE:
        from ultraquant.experiments import requests_gate as R
        report = R.run_gate()
        _CACHE["r"] = {"passes": report.passes, "reason": report.reason,
                       "measured": report.measured, "errors": report.errors}
    return _CACHE["r"]


def nothing_regresses() -> bool:
    r = _requests()
    m = r["measured"]
    return (r["passes"] and not m.get("wrong")
            and sum(m.get("hints", {}).values()) <= 68
            and len(m.get("unknown named another", [])) <= 18)


CASES = {
    "1 chat stays chat": chat_stays_chat,
    "2 no ambiguity resolved by absence": no_ambiguity_by_absence,
    "3 no held fact hidden": no_held_fact_hidden,
    "4 nothing regresses": nothing_regresses,
}


def _plants():
    from ultraquant.interpreter import thoughts as T
    from ultraquant.memory import systematic as SM
    return [
        ("P50 the committed chat asserts catalogue matches", "1 chat stays chat",
         [mock.patch.object(T.Reason, "_chat", _chat_437eedf)]),
        ("P51 the committed answer judges ambiguity among held facts",
         "2 no ambiguity resolved by absence",
         [mock.patch.object(SM.SystematicMemory, "catalogue_answer",
                            _catalogue_answer_437eedf)]),
        ("P52 the committed top-10 refusal", "3 no held fact hidden",
         [mock.patch.object(SM.SystematicMemory, "_unheld_subject",
                            _unheld_subject_437eedf)]),
    ]


@dataclass
class Review9Report:
    """Whether the ninth review's three faults are fixed.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        world_live: Whether chat's own choice misses what the catalogue finds.
        requests: §11.142's gate, rerun.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    world_live: bool = False
    requests: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(regressions: bool = True) -> Review9Report:
    report = Review9Report(passes=False)
    cases = dict(CASES)
    if not regressions:                 # the reproduction run skips case 4
        cases.pop("4 nothing regresses")
    for name, case in cases.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.world_live = world_is_live()
    except Exception as exc:
        report.errors["world"] = repr(exc)
    if "r" in _CACHE:
        report.requests = _CACHE["r"]
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = (report.world_live and len(report.planted) == 3
             and all(report.planted.values()))
    met = len(report.cases) == len(CASES) and all(report.cases.values())
    report.passes = valid and met
    failed = [n for n, ok in report.cases.items() if not ok]
    if not valid:
        report.reason = ("VOID: " + ("" if report.world_live else "the world "
                         "check failed; ") + "plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}; "
                         f"failed: {failed}")
    elif not met:
        report.reason = f"FAIL: {failed}"
    else:
        report.reason = "PASS: 4 cases; the world is live; 3 of 3 plants caught"
    return report


def main() -> int:
    import sys
    report = run_gate(regressions="--reproduce" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "world_live": report.world_live,
                      "requests": report.requests, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
