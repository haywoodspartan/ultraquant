"""The Chat/Interpreter front end.

Free text runs through the predefined thought pipeline; colon-commands drive the
machinery directly — inspecting the shard catalog and resident set, changing the
RAM budget, gating the network, triaging the contemporary stash, teaching glyphs,
and consolidating everything into a packed library plus an Ar(T)chive snapshot.

Three ways to run it::

    python -m ultraquant.interpreter.chat                 # interactive REPL
    python -m ultraquant.interpreter.chat --once "2+2?"   # single turn
    python -m ultraquant.interpreter.chat --script s.txt  # replay a script

The ``--script`` and ``--once`` forms make the whole interpreter testable without
a terminal.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

from ultraquant.interpreter.selflearn import SelfLearner
from ultraquant.interpreter.stash import StashError
from ultraquant.interpreter.thoughts import Session, build_session, run_pipeline
from ultraquant.pattern.recognition import LABELS

#: One row of a pasted glyph: exactly five cells of set/unset.
_GLYPH_ROW = re.compile(r"[#.]{5}$")

HELP = """\
UltraQuant Chat/Interpreter - commands:
  :help                  show this
  :trace                 the thought trace of the last input
  :mem                   memory statistics
  :facts [substr]        stored facts (optionally filtered)
  :shards                the shard catalog (id, category, where, bytes, uses)
  :resident              what is currently paged into RAM, and the budget
  :budget <kb>           change the resident-set budget
  :stash [id]            staged web claims, or one entry in full
  :analyze               classify and corroborate staged claims
  :promote <id> [force]  promote a staged claim to a stored fact
  :reject <id> [reason]  reject a staged claim
  :online on|off         gate network access
  :fetch <url>           fetch a URL into the contemporary stash
  :code <source>         run one line through the sandboxed code function
  :entropy [src]         judge an entropy source by its output alone
  :settings              where preferences live, and what is in them
  :correct <cat> <text>  this text should have routed to <cat>
  :learn                 survey for gaps; the model asks what it wants to know
  :learn answer <text>   answer the current question (glyph answers: 5 rows follow)
  :learn skip            set the current question aside
  :learn research        try the web first (quarantined; needs ':online on')
  :learn panel <m>...    ask local models; corroborated answers are offered
  :study [n]             n self-study cycles: survey gaps, ask the panel,
                         apply only what independent voices corroborate
  :panel                 local LM Studio models, grouped by independent lineage
  :panel <m>.. ? <q>     ask each model in isolation; agreement counted by voice
  :teach <cat> <label>   then 5 glyph rows of [#.]{5}
  :recognize             then 5 glyph rows of [#.]{5}
  :consolidate           pack hot shards into a library + snapshot
  :attach <file.uql>     attach an existing shard library
  :snapshot              commit an Ar(T)chive snapshot
  :quit                  leave
Anything else is treated as input to the thought pipeline, which
understands - among others - these forms:
  statements   "the tower material is iron" / "... is not steel"
               (a conflicting statement is revised aloud: the old
               belief named, retracted derivations counted); "the A
               material and the B material are iron" teaches every
               part
  questions    "what is the tower material?"
  chains       "what is the tower architect hometown region climate?"
               (up to four facts bridged; the trail is the answer)
  yes/no       "is the tower material iron?" - yes against belief, no
               against contrary belief, and an honest don't-know
               otherwise; absence is never no
  why          "why is the tower hardness 490 units?" - answered from
               provenance; a false premise is corrected, never
               explained
  compare      "is the tower taller than the bridge?" - taller/heavier/
               longer name their attribute themselves (units convert;
               missing operands refuse aloud; operands the store lacks
               are derived and marked)
  which        "which is the tallest?" or "which height is the tallest?"
               - ranked over every held fact, scope named, ties named,
               denials never counted; vague words (biggest) ask you to
               name the attribute
  aggregates   "how many height facts do you hold?" / "what is the
               total height?" / "what is the average height?"
  arithmetic   "what is the sum of the tower height and the bridge
               height?" (named operands; units convert or refuse)
  math         "what is 3 + 4 * 5?" / "6 * 7" / "calculate 100 / 8" -
               precedence and parentheses, evaluated over exact
               rationals: 0.1 + 0.2 is 0.3, one third prints as 1/3
               because 0.333... is a different number, and division by
               zero refuses rather than name a number
  lists        "what is the average of 3, 5 and 10?" - sum, average,
               largest, smallest over WRITTEN numbers (units convert;
               a list naming held facts goes to 'aggregates' instead)
  rounding     "what is 100 / 3 to 2 decimal places?" - exactness is
               the default and rounding is a request; the answer says
               it was rounded and names the exact value it came from
  is-it-so     "is 2 + 2 = 5?" / "is 3 * 4 greater than 10?" / "is the
               tower height * 2 greater than 500 meters?" - either side
               may be a number, a quantity, a belief or an expression,
               and a side that cannot be read refuses rather than guess
  powers       "what is 2 ^ 10?" / "what is 5 squared?" / "what is
               the square root of 16?" - whole powers exact, rational
               roots exact, and an irrational root given as proved
               bounds ("sqrt 2 is between 1.414213562 and
               1.414213563") rather than a value it does not have
  percentages  "what is 20% of 300?" / "what is 15% of the tower
               height?" - a hundredth part taken, composing with units
               and chains; "300 + 20%" refuses, because that shape is a
               convention rather than an arithmetic
  quantities   "what is the tower height times 3?" / "what is 300
               meters + 2 kilometers?" - an operand may be a number, a
               quantity, or a belief; units convert by definition, a
               length over a length is a ratio, and meters times meters
               refuses rather than invent a unit
  choices      "is the tower material steel or iron?" - the held
               disjunct answers by name; "Neither" names the actual; a
               denial rules out without electing; an unheld subject
               never picks a side
  history      "what was the tower material?" - every past belief in
               order from the revision record; a fact stated once has
               no history, none is invented, and a retracted
               conclusion names its takedown and the premise that
               caused it
  affirmation  "yes" after a derived answer consolidates it (and a
               revised premise takes its consolidations down); "yes"
               after an asserted belief confirms it as direct
               testimony (confidence 0.90 - stronger than a passing
               restatement, which reinforces by 0.1); "no" contests it
               - confidence drops to 0.30, the correction is asked
               for, and doubt never deletes; "no" after a derived
               answer declines it and names every premise so the
               wrong one can be restated
A near-key statement ("the old tower material is ...") also names the
held base it sits beside: "I separately hold: tower material is
iron" - information, never a merge."""


class ChatCLI:
    """Command loop over a :class:`Session`."""

    def __init__(self, session: Session, out=sys.stdout) -> None:
        """Bind the CLI to a session and an output stream."""
        self.session = session
        self.learner = SelfLearner(session)
        self.out = out
        self.running = True

    def emit(self, text: str) -> None:
        """Write one line of output."""
        print(text, file=self.out)

    # ------------------------------------------------------------- dispatch

    def handle(self, line: str, more: Iterator[str] | None = None) -> None:
        """Handle one input line, pulling extra lines for multi-line commands."""
        line = line.rstrip("\n")
        if not line.strip():
            return
        if not line.startswith(":"):
            # A bare glyph row starts a paste: pull the rest of the grid
            # so the pipeline sees one five-row glyph, not five unrelated
            # lines each answered "I have nothing on that".
            if _GLYPH_ROW.fullmatch(line.strip()):
                rows, leftover = self._collect_glyph(line.strip(), more)
                if len(rows) == 5:
                    response, _trace = run_pipeline("\n".join(rows),
                                                    self.session)
                    self.emit(response)
                else:
                    for row in rows:
                        response, _trace = run_pipeline(row, self.session)
                        self.emit(response)
                if leftover is not None:
                    self.handle(leftover, more)
                return
            response, _trace = run_pipeline(line, self.session)
            self.emit(response)
            return

        parts = line[1:].split()
        command, args = (parts[0].lower() if parts else ""), parts[1:]
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            self.emit(f"Unknown command ':{command}'. Try ':help'.")
            return
        try:
            handler(args, more)
        except Exception as exc:  # noqa: BLE001 - the REPL must survive anything
            self.emit(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _collect_glyph(first: str,
                       more: Iterator[str] | None
                       ) -> tuple[list[str], str | None]:
        """Gather up to five glyph rows starting from ``first``.

        Stops early on a blank line (the interactive escape hatch -
        input() cannot raise StopIteration, so a user who typed one
        stray row gets out by pressing enter), on a line that is not a
        glyph row (returned as leftover for normal handling), or when
        the source runs dry.
        """
        rows = [first]
        while len(rows) < 5:
            if more is not None:
                try:
                    row = next(more)
                except StopIteration:
                    break
            else:
                row = input()
            stripped = row.strip()
            if not stripped:
                break
            if not _GLYPH_ROW.fullmatch(stripped):
                return rows, row
            rows.append(stripped)
        return rows, None

    @staticmethod
    def _take_rows(more: Iterator[str] | None, count: int = 5) -> list[str]:
        """Read ``count`` glyph rows from the input source."""
        rows: list[str] = []
        while len(rows) < count:
            if more is not None:
                try:
                    row = next(more)
                except StopIteration:
                    break
            else:
                row = input()
            row = row.strip()
            if row:
                rows.append(row)
        return rows

    # ------------------------------------------------------------- commands

    def _cmd_help(self, args: list[str], more) -> None:
        """Print the command list."""
        self.emit(HELP)

    def _cmd_quit(self, args: list[str], more) -> None:
        """Leave the loop."""
        self.running = False
        self.emit("Consolidating and exiting.")

    def _cmd_trace(self, args: list[str], more) -> None:
        """Show the last thought trace."""
        trace = self.session.last_trace
        if not trace:
            self.emit("No trace yet.")
            return
        for step in trace:
            self.emit(f"  {step['thought']:<9} {step['summary']}")

    def _cmd_mem(self, args: list[str], more) -> None:
        """Show memory statistics."""
        self.emit(str(self.session.memory.stats()))

    def _cmd_facts(self, args: list[str], more) -> None:
        """List stored facts."""
        needle = " ".join(args).lower()
        memory = self.session.memory
        rows = [
            (k, v) for k, v in sorted(
                (key, memory.recall_fact(key) or {}) for key in memory.fact_keys()
            )
            if not needle or needle in k.lower() or needle in str(v.get("value", "")).lower()
        ]
        if not rows:
            self.emit("No facts stored." if not needle else f"No facts matching {needle!r}.")
            return
        for key, fact in rows:
            self.emit(f"  {key} = {fact['value']}  (conf {fact['confidence']:.2f}, "
                      f"x{fact.get('reinforcements', 0)})")

    def _cmd_shards(self, args: list[str], more) -> None:
        """Show the shard catalog."""
        catalog = self.session.vault.catalog()
        if not catalog:
            self.emit("Vault is empty.")
            return
        self.emit(f"  {'shard':<28} {'category':<12} {'where':<8} {'bytes':>8} {'uses':>5}")
        for entry in catalog:
            self.emit(
                f"  {entry['shard_id']:<28} {entry['category']:<12} "
                f"{entry['location']:<8} {entry['nbytes']:>8} {entry['access_count']:>5}"
            )
        stats = self.session.vault.stats()
        self.emit(f"  total {stats['total_bytes']} bytes across {stats['shards']} shard(s); "
                  f"{len(stats['libraries'])} librar(y/ies)")

    def _cmd_resident(self, args: list[str], more) -> None:
        """Show the resident set and budget."""
        stats = self.session.cache.stats()
        self.emit(
            f"  budget {stats['budget_bytes']} B | resident {stats['current_bytes']} B "
            f"| peak {stats['peak_bytes']} B"
        )
        self.emit(f"  hits {stats['hits']} misses {stats['misses']} evictions {stats['evictions']}")
        vram = getattr(self.session.experts, "vram", None)
        if vram is not None:
            v = vram.stats()
            self.emit(
                f"  vram: {len(v['resident'])} layer(s) resident "
                f"({v['resident_bytes']} B of {v['budget_bytes']} B), "
                f"hits {v['hits']} misses {v['misses']} "
                f"evictions {v['evictions']}")
        window = getattr(self.session, "context", None)
        if window is not None:
            ctx = window.stats()
            self.emit(
                f"  context window: {ctx['resident_turns']} turn(s) resident "
                f"({ctx['resident_bytes']} B of {ctx['budget_bytes']} B), "
                f"{ctx['turns']} on disk ({ctx['stored_bytes']} B), "
                f"index {ctx['index_bytes']} B")
        self.emit(f"  paged in (LRU->MRU): {', '.join(stats['resident']) or '(nothing)'}")

    def _cmd_budget(self, args: list[str], more) -> None:
        """Change the resident-set budget."""
        if not args:
            self.emit("Usage: :budget <kilobytes>")
            return
        self.session.cache.set_budget(int(float(args[0]) * 1024))
        self._cmd_resident([], more)

    def _cmd_stash(self, args: list[str], more) -> None:
        """List staged claims, or show one in detail."""
        stash = self.session.stash
        if args:
            entry = stash.get(int(args[0]))
            for key in ("id", "claim", "classification", "status", "sources", "url", "notes"):
                self.emit(f"  {key}: {entry[key]}")
            return
        entries = stash.entries()
        if not entries:
            self.emit("Stash is empty - nothing fetched yet.")
            return
        for entry in entries:
            self.emit(
                f"  [{entry['id']}] {entry['classification']}/{entry['status']} "
                f"({len(entry['sources'])} source(s)): {entry['claim'][:90]}"
            )
        self.emit(f"  {stash.stats()}")

    def _cmd_analyze(self, args: list[str], more) -> None:
        """Re-run stash analysis."""
        self.emit(str(self.session.stash.analyze(self.session.memory)))

    def _cmd_promote(self, args: list[str], more) -> None:
        """Promote a staged claim into memory."""
        if not args:
            self.emit("Usage: :promote <id> [force]")
            return
        force = len(args) > 1 and args[1].lower() == "force"
        try:
            key = self.session.stash.promote(int(args[0]), self.session.memory, force=force)
        except StashError as exc:
            self.emit(f"Refused: {exc}")
            return
        self.session.memory.save()
        self.emit(f"Promoted to fact: {key}")

    def _cmd_reject(self, args: list[str], more) -> None:
        """Reject a staged claim."""
        if not args:
            self.emit("Usage: :reject <id> [reason]")
            return
        self.session.stash.reject(int(args[0]), " ".join(args[1:]))
        self.emit(f"Rejected entry {args[0]}.")

    def _cmd_online(self, args: list[str], more) -> None:
        """Gate network access."""
        if args and args[0].lower() in ("on", "off"):
            self.session.web.set_online(args[0].lower() == "on")
        self.emit(f"Web access is {'ON' if self.session.web.online else 'OFF'}.")

    def _cmd_fetch(self, args: list[str], more) -> None:
        """Fetch a URL into the stash."""
        if not args:
            self.emit("Usage: :fetch <url>")
            return
        response, _trace = run_pipeline(args[0], self.session)
        self.emit(response)

    def _cmd_code(self, args: list[str], more) -> None:
        """Run source through the sandboxed code function."""
        response, _trace = run_pipeline("code: " + " ".join(args), self.session)
        self.emit(response)

    def _cmd_teach(self, args: list[str], more) -> None:
        """Teach a glyph to a category expert."""
        if len(args) < 2:
            self.emit("Usage: :teach <category> <label>  (then 5 rows of [#.]{5})")
            return
        category, label = args[0], args[1]
        rows = self._take_rows(more)
        if len(rows) != 5:
            self.emit("Need exactly 5 glyph rows.")
            return
        labels = list(LABELS)
        if label not in labels:
            labels = sorted(set(labels + [label]))
        self.session.experts.ensure_expert(category, labels)
        report = self.learner.teach_glyph(category, labels, rows, label)
        self.emit(f"Learned '{label}' for {category}: "
                  f"loss {report['loss']:.4f}, train accuracy {report['accuracy']:.2f}")

    def _cmd_learn(self, args: list[str], more) -> None:
        """Learning mode: the model surveys its own gaps and asks.

        ``:learn`` (or ``:learn find``) runs the survey and shows the first
        question; ``:learn answer <text>`` applies an answer and shows the
        next; ``:learn skip`` sets one aside. A question that expects a glyph
        reads five rows, exactly as ``:teach`` does.

        The GUI and TUI have had this since learning mode existed; the chat
        CLI only gained it when the loop was actually driven end to end here
        and the gap showed — every surface must reach every capability.
        """
        from ultraquant.interpreter.learning import LearningSession

        verb = args[0].lower() if args else "find"
        if verb == "find":
            self.learning = LearningSession(self.session)
            questions = self.learning.survey()
            if not questions:
                self.emit("No gaps found - nothing worth asking right now.")
                return
            kinds: dict[str, int] = {}
            for question in questions:
                kinds[question.kind] = kinds.get(question.kind, 0) + 1
            summary = ", ".join(f"{count} {kind}" for kind, count in
                                sorted(kinds.items()))
            self.emit(f"{len(questions)} question(s) worth asking: {summary}")
            self._show_learn_question()
            return

        learning = getattr(self, "learning", None)
        if learning is None:
            self.emit("Run ':learn' first to survey for gaps.")
            return
        question = learning.next_question()
        if question is None:
            self.emit("Nothing is queued - run ':learn' again.")
            return
        if verb == "skip":
            learning.skip(question.id)
            self.emit(f"Skipped [{question.kind}].")
            self._show_learn_question()
            return
        if verb == "research":
            reports = learning.research()
            if not reports:
                self.emit("Nothing researchable is pending - research covers "
                          "unknown terms and topic categories.")
                return
            for report in reports:
                self.emit(f"  {report['subject']:<24} {report['outcome']:<9} "
                          f"{report['detail']}")
            self._show_learn_question()
            return
        if verb == "panel":
            self._learn_from_panel(learning, question, args[1:])
            return
        if verb in ("answer", "a"):
            if question.expects == "glyph":
                rows = self._take_rows(more)
                if len(rows) != 5:
                    self.emit("Need exactly 5 glyph rows.")
                    return
                reply = "\n".join(rows)
            else:
                reply = " ".join(args[1:]).strip()
                if not reply:
                    self.emit("Give an answer: ':learn answer <text>'.")
                    return
            answer = learning.answer(question, reply)
            mark = "learned" if answer.accepted else "not applied"
            self.emit(f"{mark}: {answer.detail}")
            self._show_learn_question()
            return
        self.emit("Usage: :learn [find|answer <text>|skip|research|"
                  "panel <model>...]")

    def _learn_from_panel(self, learning, question, models: list[str]) -> None:
        """Answer the model's own open question with a teacher panel.

        This is the LLMLS in its intended role: the system decides what it
        does not know, and several independently-lineaged local models are
        asked. It is the same shape as ``:learn research``, with a panel in
        place of the web — and the same rule, which is that a source does not
        become a fact by being asked.

        The answer is **not** applied directly. It is quarantined, and only
        offered for application when independent voices corroborated it; a
        single voice, or a split, leaves the question open. That is the whole
        difference between a panel and an oracle, and skipping it here would
        undo the accounting the panel exists to do.
        """
        from ultraquant.interpreter.llmls import LMStudioUnavailable, TeacherPanel

        if question.expects == "glyph":
            self.emit("  A glyph question needs pixels, which a text panel "
                      "cannot supply. Use ':learn answer' with 5 rows.")
            return
        if not models:
            self.emit("Usage: :learn panel <model> [<model>...]  "
                      "(see ':panel' for the catalogue)")
            return
        try:
            panel = TeacherPanel(models)
            self.emit(panel.independence_report())
            self.emit(f"  asking: {question.prompt}")
            result = panel.teach(
                question.prompt, self.session.stash,
                usages=question.context.get("usages"))
        except LMStudioUnavailable as exc:
            self.emit(f"  LM Studio unavailable: {exc}")
            return

        consensus = result["consensus"]
        self.emit(consensus.as_text())
        self.emit(f"  {result['filed']} claim(s) quarantined in the stash.")
        if consensus.corroborated:
            # Offered as the exact command, which is this surface's version of
            # the GUI pre-filling its answer box: one keystroke from applying,
            # and still requiring that keystroke.
            agreed = max(consensus.split.items(), key=lambda kv: len(kv[1]))[0]
            self.emit(f"  -> {consensus.voices} independent voices agree. "
                      "To apply it:")
            self.emit(f"       :learn answer {agreed}")
            self.emit("     or review it first with ':stash'.")
        else:
            self.emit("  -> not corroborated across independent voices; the "
                      "question stays open. This is the correct outcome, not "
                      "a failure of the panel.")

    def _show_learn_question(self) -> None:
        """Print the question at the head of the queue."""
        learning = getattr(self, "learning", None)
        question = learning.next_question() if learning else None
        if question is None:
            self.emit("No more questions - run ':learn' again later.")
            return
        self.emit(f"[{question.kind}] {question.prompt}")
        if question.options:
            self.emit("  options: " + " / ".join(question.options))
        if question.expects == "glyph":
            self.emit("  answer with ':learn answer' then 5 rows of [#.]{5}")
        else:
            self.emit("  answer with ':learn answer <text>', or ':learn skip'")

    def _cmd_entropy(self, args: list[str], more) -> None:
        """Judge an entropy source by its output alone - the black box.

        ``:entropy`` assesses the OS pool; ``:entropy jitter`` this machine's
        timing wobble. Nothing about a source's internals is consulted, which
        is the point: a plausible story about where randomness comes from
        cannot buy trust, only measured output can.
        """
        import os as _os

        from ultraquant.quantum.entropy_blackbox import assess

        which = (args[0].lower() if args else "urandom")
        if which == "jitter":
            import time as _time

            def source(n: int) -> bytes:
                deltas = []
                for _ in range(n):
                    start = _time.perf_counter_ns()
                    churn: dict[str, list[int]] = {}
                    for i in range(60):
                        churn[str(i)] = [i] * 7
                    deltas.append(_time.perf_counter_ns() - start)
                floor = min(deltas)
                return bytes(((d - floor) // 100) & 0xFF for d in deltas)
        elif which == "urandom":
            def source(n: int) -> bytes:
                return _os.urandom(n)
        else:
            self.emit("Usage: :entropy [urandom|jitter]")
            return

        verdict = assess(source, sample_bytes=4096, draws=2)
        self.emit(f"  source     : {which}")
        self.emit(f"  credited   : {verdict.credited_bits_per_byte:.2f} bits/byte "
                  f"(of 8 ideal)")
        self.emit(f"  trustworthy: {verdict.trustworthy}")
        self.emit(f"  reason     : {verdict.reason}")
        for name, score in sorted(verdict.tests.items()):
            self.emit(f"    {name:<28} {score:.3f}")

    def _cmd_study(self, args: list[str], more) -> None:
        """Run self-study cycles: the library closing its own gaps."""
        from ultraquant.interpreter.study import StudyCycle

        try:
            cycles = max(1, int(args[0])) if args else 1
        except ValueError:
            self.emit("Usage: :study [n]")
            return
        models = self._study_panel_models()
        if not models:
            self.emit("No panel models reachable - ':study' needs LM "
                      "Studio. Gaps stay open; ':learn' still works.")
            return
        self.emit(f"studying with panel: {', '.join(models)}")
        cycle = StudyCycle(self.session, panel_models=models)
        for index in range(cycles):
            report = cycle.run()
            self.emit(report.as_text())
            if report.errors or not report.asked:
                break

    @staticmethod
    def _study_panel_models() -> list[str]:
        """One model per independent voice, smallest member of each."""
        try:
            from ultraquant.forge.train_from_llm import parameter_count
            from ultraquant.interpreter.llmls import (catalogue,
                                                      independent_groups)

            cards = [c for c in catalogue() if c.is_chat]
            groups = independent_groups(cards)
            chosen = []
            for group in groups:
                members = sorted(group, key=lambda c:
                                 (parameter_count(c.id), c.id))
                chosen.append(members[0].id)
            return chosen
        except Exception:  # noqa: BLE001 - no LM Studio, no panel
            return []

    def _cmd_panel(self, args: list[str], more) -> None:
        """Interrogate a panel of local LM Studio models - the LLMLS.

        ``:panel`` lists the catalogue with its independence accounting;
        ``:panel <model> <model> ... ? <question>`` puts one question to each
        named model in isolation and reports what they agreed on, weighted by
        *independent lineage* rather than by headcount.

        Nothing the panel says is believed. Agreement across independent
        voices is evidence; agreement between a model and its own fine-tune is
        not evidence at all, and the report distinguishes them. Use ``:stash``
        and ``:promote`` to decide what, if anything, becomes a fact.
        """
        from ultraquant.interpreter.llmls import (
            LMStudioUnavailable, TeacherPanel, catalogue, independent_groups,
        )

        try:
            if not args:
                cards = catalogue()
                chat = [card for card in cards if card.is_chat]
                groups = independent_groups(chat)
                self.emit(f"  {len(chat)} chat model(s) in {len(groups)} "
                          f"independent voice(s):")
                for index, group in enumerate(groups, 1):
                    self.emit(f"    voice {index} (arch={group[0].arch or '?'}, "
                              f"publisher={group[0].publisher or '?'}):")
                    for card in group:
                        mark = "*" if card.loaded else " "
                        self.emit(f"      {mark} {card.id}")
                self.emit("  * = currently loaded. Voice counts are a LOWER "
                          "BOUND on correlation.")
                return

            if "?" not in args:
                self.emit("Usage: :panel  |  :panel <model> [<model>...] "
                          "? <question>")
                return
            cut = args.index("?")
            models, question = args[:cut], " ".join(args[cut + 1:]).strip()
            if not models or not question:
                self.emit("Need at least one model and a question.")
                return

            panel = TeacherPanel(models)
            self.emit(panel.independence_report())
            self.emit("  asking (local models are slow; this may take a while)...")
            consensus = panel.ask(question, max_tokens=120)
            self.emit(consensus.as_text())
            if consensus.corroborated:
                self.emit("  -> corroborated across independent voices; "
                          "still quarantined until promoted.")
            else:
                self.emit("  -> NOT corroborated. One voice is one source, "
                          "however many models back it.")
        except LMStudioUnavailable as exc:
            self.emit(f"  LM Studio unavailable: {exc}")

    def _cmd_settings(self, args: list[str], more) -> None:
        """Show the persistent settings, and where they are kept.

        Read-only here on purpose. The chat CLI takes its budget and network
        state from flags and colon-commands for the life of one run; the GUI
        and TUI are the surfaces that hold a session open long enough for a
        preference to be worth remembering. This exists so the file is
        discoverable from every surface, not just the two that write it.
        """
        from ultraquant.config import Settings

        self.emit(Settings.load().describe())

    def _cmd_correct(self, args: list[str], more) -> None:
        """Tell the router a query belonged somewhere else.

        ``:correct arithmetic add these two numbers``

        The only ground truth for a routing error is a person saying so, and
        there was no way to say it. ``:learn`` cannot do this job: it reinforces
        from the *reply* text rather than the misrouted query, and none of its
        question kinds surfaces a misroute. Measured on the deployed library, a
        fact answered about arithmetic moved it 1.2 -> 1.4 while the wrong
        winner stayed at 2.15.

        This strengthens the query's own tokens for the right category and
        weakens whichever category was winning them, which is what actually
        reverses an entrenched decision rather than slowly out-climbing it.
        """
        if len(args) < 2:
            self.emit("Usage: :correct <category> <the text that routed wrongly>")
            return
        category, text = args[0], " ".join(args[1:])
        router = self.session.router
        if category not in router._base:
            self.emit(f"Unknown category {category!r}. Known: "
                      + ", ".join(sorted(router._base)[:12]))
            return
        before = router.route(text, top_k=2)
        result = router.correct(text, category)
        if not result["tokens"]:
            self.emit("Nothing to learn from that text - it is all stopwords.")
            return
        after = router.route(text, top_k=2)
        self.session.save()
        self.emit(f"  tokens     : {', '.join(result['tokens'])}")
        self.emit(f"  before     : {before}")
        self.emit(f"  after      : {after}")
        if result["weakened"]:
            self.emit("  weakened   : " + ", ".join(
                f"{name} -{amount}" for name, amount in
                sorted(result["weakened"].items())))
        landed = after and after[0][0] == category
        self.emit(f"  -> {'corrected' if landed else 'still not winning; '
                          'correct it again or check the category'}")

    def _cmd_recognize(self, args: list[str], more) -> None:
        """Recognize a glyph."""
        rows = self._take_rows(more)
        if len(rows) != 5:
            self.emit("Need exactly 5 glyph rows.")
            return
        response, _trace = run_pipeline("\n".join(rows), self.session)
        self.emit(response)

    def _cmd_consolidate(self, args: list[str], more) -> None:
        """Pack hot shards and snapshot."""
        report = self.learner.consolidate()
        self.emit(
            f"Packed {report['packed']} shard(s)"
            + (f" into {report['library']}" if report["library"] else "")
            + (f"; snapshot {report['snapshot']}" if report["snapshot"] else "")
        )

    _cmd_pack = _cmd_consolidate

    def _cmd_attach(self, args: list[str], more) -> None:
        """Attach an existing shard library."""
        if not args:
            self.emit("Usage: :attach <file.uql>")
            return
        count = self.session.vault.attach(args[0])
        self.emit(f"Attached {count} shard(s) from {args[0]} (index only - no payloads read).")

    def _cmd_snapshot(self, args: list[str], more) -> None:
        """Commit an Ar(T)chive snapshot."""
        if self.session.archive is None:
            self.emit("No archive configured.")
            return
        version = self.session.archive.commit(
            "manual",
            {
                "vault": self.session.vault.stats(),
                "memory": self.session.memory.stats(),
                "cache": self.session.cache.stats(),
                "stash": self.session.stash.stats(),
            },
        )
        self.emit(f"Committed snapshot {version}.")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="UltraQuant Chat/Interpreter")
    parser.add_argument("--root", default="./uq_home", help="session directory")
    parser.add_argument("--online", action="store_true", help="enable web access at start")
    parser.add_argument("--budget-kb", type=int, default=1024, help="resident shard budget")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--script", help="replay a file of inputs instead of reading stdin")
    parser.add_argument("--once", help="run one input and exit")
    args = parser.parse_args(argv)

    # A legacy Windows console is cp1252 and cannot encode every character a
    # fetched page might contain. Degrade those to '?' rather than killing the
    # session with a UnicodeEncodeError mid-command.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream without reconfigure
        pass

    from ultraquant.config import Settings

    semantic = bool(Settings.load().get("lmstudio.semantic_suggest", True))
    session = build_session(
        Path(args.root),
        budget_bytes=args.budget_kb * 1024,
        online=args.online,
        seed=args.seed,
        semantic=semantic,
    )
    cli = ChatCLI(session)

    if args.once is not None:
        response, _trace = run_pipeline(args.once, session)
        print(response)
        return 0

    if args.script:
        lines = Path(args.script).read_text(encoding="utf-8").splitlines()
        stream = iter(lines)
        for line in stream:
            if not line.strip():
                continue
            print(f"> {line}")
            cli.handle(line, stream)
            if not cli.running:
                break
        return 0

    print("UltraQuant Chat/Interpreter. ':help' for commands, ':quit' to leave.")
    while cli.running:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cli.handle(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
