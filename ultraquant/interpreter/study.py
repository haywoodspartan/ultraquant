"""The study cycle: the library closing its own gaps, one pass at a time.

Every piece of self-directed learning exists separately — the gap survey
(learning.py), the teacher panel with containment corroboration
(§11.41), the quarantine, curiosity (§11.33), application through the
answer handlers. This module is the loop that runs them in one pass,
which is the self-learning claim in its mature form: the system decides
what it does not know, asks, believes only what independent voices
corroborate, and files the rest as open.

The trust order inside a cycle is the standing one: web research first
when online (found-is-not-believed quarantine), the panel second, the
human always able to preempt both by just answering ':learn'. What a
cycle may APPLY is exactly what the fixed training driver applies —
``outcome.accepted`` after corroboration, untrained-category answers
formatted as statements — and everything applied is reported for veto.

The risk this module adds is hallucination-through-corroboration:
independent voices confidently agreeing on an invented thing. The study
gate measures exactly that with fake-term decoys before the loop counts
as a capability, and the panel channel's answers stay quarantined claims
regardless — application stores a fact, the stash keeps the provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["StudyCycle", "StudyReport"]

#: Question kinds a cycle may try to close without a human.
_TEXT_KINDS = ("unknown-term", "untrained-category", "missing-premise")


@dataclass
class StudyReport:
    """What one cycle did, in full, for the human's veto.

    Attributes:
        surveyed: How many questions the survey found.
        asked: Questions actually put to a channel.
        applied: ``(subject, answer, voices)`` for every applied fact.
        quarantined: Claims filed in the stash across the cycle.
        open: Subjects that stay open, with the reason.
        errors: Channel failures, per subject.
    """

    surveyed: int = 0
    asked: int = 0
    applied: list = field(default_factory=list)
    quarantined: int = 0
    open: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)

    def as_text(self) -> str:
        """One cycle, one honest paragraph."""
        lines = [f"study cycle: {self.surveyed} question(s) surveyed, "
                 f"{self.asked} asked, {len(self.applied)} applied, "
                 f"{self.quarantined} claim(s) quarantined"]
        for subject, answer, voices in self.applied:
            lines.append(f"  APPLIED {subject!r} = {answer[:60]!r} "
                         f"({voices} voices)")
        for subject, why in self.open:
            lines.append(f"  open: {subject!r} - {why}")
        for subject, why in self.errors.items():
            lines.append(f"  error: {subject!r} - {why}")
        return "\n".join(lines)


class StudyCycle:
    """One self-directed learning pass over a session.

    Args:
        session: The live :class:`~ultraquant.interpreter.thoughts.Session`.
        panel_models: Model ids for the teacher panel, one per independent
            voice ideally. None means panel questions stay open.
        max_questions: Budget per cycle; the queue is score-ordered, so
            this spends the budget on the most valuable gaps.
    """

    def __init__(self, session: Any, panel_models: list[str] | None = None,
                 max_questions: int = 4) -> None:
        self.session = session
        self.panel_models = list(panel_models or [])
        self.max_questions = int(max_questions)
        self._panel = None

    # ------------------------------------------------------------- channels

    def _panel_for(self):
        """The TeacherPanel, constructed once, or None."""
        if not self.panel_models:
            return None
        if self._panel is None:
            from ultraquant.interpreter.llmls import (LMStudioUnavailable,
                                                      TeacherPanel)

            try:
                self._panel = TeacherPanel(self.panel_models)
            except LMStudioUnavailable:
                return None
        return self._panel

    def _ask_panel(self, learning, question) -> tuple[str, Any] | None:
        """(agreed_answer, consensus) when corroborated, else None."""
        panel = self._panel_for()
        if panel is None:
            return None
        from ultraquant.interpreter.llmls import LMStudioUnavailable

        try:
            result = panel.teach(question.prompt, self.session.stash,
                                 usages=question.context.get("usages"))
        except LMStudioUnavailable as exc:
            raise RuntimeError(str(exc)) from exc
        consensus = result["consensus"]
        self._last_filed = result["filed"]
        if not consensus.corroborated:
            return None
        agreed = max(consensus.split.items(), key=lambda kv: len(kv[1]))[0]
        return agreed, consensus

    # ------------------------------------------------------------ the cycle

    def run(self) -> StudyReport:
        """Survey, ask, apply what corroborates, report everything.

        Returns:
            The :class:`StudyReport`.
        """
        from ultraquant.interpreter.learning import LearningSession

        report = StudyReport()
        learning = LearningSession(self.session)
        questions = learning.survey()
        report.surveyed = len(questions)

        askable = [q for q in questions if q.kind in _TEXT_KINDS
                   and q.expects == "text"][:self.max_questions]
        for question in askable:
            report.asked += 1
            self._last_filed = 0
            try:
                answer = self._ask_panel(learning, question)
            except RuntimeError as exc:
                report.errors[question.subject] = str(exc)
                break
            report.quarantined += getattr(self, "_last_filed", 0)
            if answer is None:
                report.open.append((question.subject,
                                    "not corroborated across voices"))
                continue
            agreed, consensus = answer
            applied_text = agreed
            if question.kind == "untrained-category":
                applied_text = f"{question.subject} is {agreed}"
            outcome = learning.answer(question, applied_text)
            if outcome.accepted:
                report.applied.append((question.subject, applied_text,
                                       consensus.voices))
            else:
                report.open.append((question.subject,
                                    f"handler refused: {outcome.detail}"))

        if self.session.memory.shards is not None:
            self.session.memory.shards.flush()
        return report
