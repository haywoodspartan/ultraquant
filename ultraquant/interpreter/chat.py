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
import sys
from pathlib import Path
from typing import Iterator

from ultraquant.interpreter.selflearn import SelfLearner
from ultraquant.interpreter.stash import StashError
from ultraquant.interpreter.thoughts import Session, build_session, run_pipeline
from ultraquant.pattern.recognition import LABELS

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
  :learn                 survey for gaps; the model asks what it wants to know
  :learn answer <text>   answer the current question (glyph answers: 5 rows follow)
  :learn skip            set the current question aside
  :learn research        try the web first (quarantined; needs ':online on')
  :teach <cat> <label>   then 5 glyph rows of [#.]{5}
  :recognize             then 5 glyph rows of [#.]{5}
  :consolidate           pack hot shards into a library + snapshot
  :attach <file.uql>     attach an existing shard library
  :snapshot              commit an Ar(T)chive snapshot
  :quit                  leave
Anything else is treated as input to the thought pipeline."""


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
        self.emit("Usage: :learn [find|answer <text>|skip|research]")

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

    session = build_session(
        Path(args.root),
        budget_bytes=args.budget_kb * 1024,
        online=args.online,
        seed=args.seed,
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
