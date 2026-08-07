"""Terminal UI: everything the desktop app does, over SSH.

The GUI needs Tkinter and a display. Neither is present on a headless Linux box,
a container, or a machine reached over SSH — which is where a model library of
this size actually tends to live. This is the same eight surfaces driven from a
terminal.

**It is deliberately not curses.** ``curses`` ships with CPython on Linux and
macOS and *not* on Windows, so a curses TUI would run on exactly the platforms
that already had options and fail on the one that does not. This renders with
plain writes and optional ANSI, so it works on a Windows console, an xterm, a
serial console, a CI log, and a pipe.

That choice has a second payoff: every screen is a function from a command to
text, so the whole interface is testable without a terminal at all. ``handle()``
takes a line and returns what would be printed; ``run()`` is only the loop around
it.

Run it::

    python -m ultraquant.tui
    ./ultraquant.sh tui
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

__all__ = ["UltraQuantTUI", "main"]

#: The eight surfaces, mirroring the desktop app's tabs.
SCREENS: tuple[tuple[str, str], ...] = (
    ("chat", "talk to the model, see the thought trace"),
    ("learn", "let the model find its gaps and ask you, or a model panel"),
    ("compute", "CPU / GPU tiers, threads, RAM budget"),
    ("forge", "build a model library from scratch"),
    ("library", "the shard catalog and what is resident"),
    ("storage", "where the library lives; library and forge locations"),
    ("stash", "triage web claims: fact, opinion, or drop"),
    ("panel", "ask local LM Studio models, counted by independent lineage"),
    ("bench", "measure the execution tiers on this machine"),
)


def _supports_ansi(stream) -> bool:
    """Whether it is safe to emit colour and cursor codes.

    A pipe, a CI log or a redirected file gets none, which is what keeps the
    output readable when this is captured rather than watched.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.name == "nt":
        # Windows 10+ consoles understand VT sequences once the mode is set;
        # older ones would print the escapes literally, so ask first.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
        except Exception:  # noqa: BLE001 - any failure means "assume not"
            return False
    return os.environ.get("TERM", "") not in ("", "dumb")


class UltraQuantTUI:
    """The terminal interface.

    Args:
        home: Session root; defaults to ``./uq_home``.
        stream: Where to write. Injectable so tests can capture it.
        color: Force ANSI on or off; ``None`` auto-detects.
    """

    def __init__(
        self, home: str | os.PathLike | None = None,
        stream: Any | None = None, color: bool | None = None,
    ) -> None:
        self.home = Path(home or Path.cwd() / "uq_home")
        self.stream = stream if stream is not None else sys.stdout
        self.color = _supports_ansi(self.stream) if color is None else bool(color)
        self.screen = "chat"
        self.session: Any = None
        self.learner: Any = None
        self.running = True
        self._last_trace: list[dict] = []

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #

    def _paint(self, text: str, code: str) -> str:
        """Wrap ``text`` in an ANSI colour, or return it unchanged."""
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def width(self) -> int:
        """Terminal width, with a sane default when it cannot be known."""
        try:
            return max(40, min(120, shutil.get_terminal_size((100, 30)).columns))
        except Exception:  # noqa: BLE001 - not worth failing over
            return 100

    def write(self, text: str = "") -> None:
        """Emit a line.

        Every message this class produces is ASCII on purpose: a Windows console
        still defaults to cp1252, and one stray em-dash becomes a replacement
        character in the middle of a sentence. Text that came from the model or
        from a file is not ours to sanitise, so the stream is also asked to
        substitute rather than raise on anything it cannot encode.
        """
        try:
            self.stream.write(text + "\n")
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "ascii"
            self.stream.write(
                (text + "\n").encode(encoding, errors="replace").decode(encoding)
            )

    def banner(self) -> str:
        """The header line: where we are and what is under us."""
        tabs = []
        for name, _help in SCREENS:
            tabs.append(self._paint(f"[{name}]", "1;36") if name == self.screen
                        else f" {name} ")
        return "".join(tabs)

    def status(self) -> str:
        """Detected accelerators and session state, as one line."""
        bits = []
        try:
            from ultraquant.native.dispatch import describe_tiers

            bits.append(describe_tiers())
        except Exception:  # noqa: BLE001 - detection must never break the UI
            try:
                from ultraquant.native import accel

                bits.append("C++" if accel.load_cpu() else "pure-python")
                if accel.load_gpu():
                    bits.append("CUDA")
            except Exception:  # noqa: BLE001
                bits.append("tiers unknown")
        if self.session is not None:
            try:
                stats = self.session.vault.stats()
                bits.append(f"{stats.get('shards', 0)} shards")
                bits.append(f"{len(self.session.cache.stats()['resident'])} resident")
            except Exception:  # noqa: BLE001
                pass
        return " | ".join(str(b) for b in bits if b)

    # ------------------------------------------------------------------ #
    # session
    # ------------------------------------------------------------------ #

    def ensure_session(self) -> Any:
        """Open the session on first use."""
        if self.session is None:
            from ultraquant.interpreter.thoughts import build_session

            self.session = build_session(self.home, seed=0)
        return self.session

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #

    def handle(self, line: str) -> str:
        """Run one input line and return what should be shown.

        Every screen is reachable from every other, so a colon command is
        checked before the current screen sees the text.
        """
        line = line.strip()
        if not line:
            return ""
        if line.startswith(":"):
            return self._command(line[1:].strip())
        handler: Callable[[str], str] = getattr(
            self, f"_screen_{self.screen}", self._screen_chat
        )
        try:
            return handler(line)
        except Exception as exc:  # noqa: BLE001 - a bad command must not exit
            return self._paint(f"error: {exc}", "31") + "\n" + traceback.format_exc(
                limit=2
            )

    def _command(self, command: str) -> str:
        """Handle a ``:`` command."""
        name, _, rest = command.partition(" ")
        name = name.lower()
        rest = rest.strip()
        if name in ("q", "quit", "exit"):
            self.running = False
            return "bye"
        if name in ("h", "help", "?"):
            return self.help_text()
        if name == "status":
            return self.status()
        if name == "home":
            if rest:
                self.home = Path(rest)
                self.session = None
                self.learner = None
                return f"home is now {self.home} (session will reopen on next use)"
            return str(self.home)
        if name == "trace":
            if not self._last_trace:
                return "no trace yet"
            return "\n".join(
                f"  {step['thought']:<9} {step.get('summary', '')}"
                for step in self._last_trace
            )
        for screen, _help in SCREENS:
            if name == screen:
                self.screen = screen
                return (self._screen_enter(screen) if rest == "" else
                        self.handle(rest))
        return f"unknown command ':{name}' - try :help"

    def help_text(self) -> str:
        """The help screen."""
        lines = ["screens (switch with :<name>, or ':<name> <command>')"]
        for name, blurb in SCREENS:
            lines.append(f"  :{name:<9} {blurb}")
        lines += [
            "",
            "anywhere",
            "  :status     detected tiers and session state",
            "  :trace      the thought trace of the last chat turn",
            "  :home <dir> point at a different session root",
            "  :help  :quit",
        ]
        return "\n".join(lines)

    #: What each screen shows when you arrive on it, so ':library' lists the
    #: catalog rather than only explaining that it could.
    DEFAULT_VIEW = {
        "learn": "", "compute": "detect", "forge": "",
        "library": "list", "storage": "show", "stash": "list", "bench": "",
    }

    def _screen_enter(self, screen: str) -> str:
        """Header, hint, and the screen's default view."""
        blurb = dict(SCREENS)[screen]
        hint = {
            "chat": "type anything; a 5x5 glyph of # and . is read as a pattern",
            "learn": "'find' to survey gaps; 'research' tries the web; 'skip' to pass",
            "compute": "'detect' to probe tiers, 'ram <MB>' to set the budget",
            "forge": "'build [n]' to forge a library, 'compare' to time the tiers",
            "library": "'list' the catalog, 'resident' what is in RAM, 'parts'",
            "storage": "'show', 'use <uri>', 'library <dir>', 'forge <dir>'",
            "stash": "'list', 'promote <id>', 'reject <id>'",
            "bench": "'run' to measure the tiers, 'paging' for the shard demo",
        }.get(screen, "")
        header = f"-- {screen}: {blurb}\n   {hint}"
        view = self.DEFAULT_VIEW.get(screen, "")
        if not view:
            return header
        try:
            body = getattr(self, f"_screen_{screen}")(view)
        except Exception as exc:  # noqa: BLE001 - arriving must never fail
            body = f"({exc})"
        return header + ("\n" + body if body else "")

    # ------------------------------------------------------------------ #
    # screens
    # ------------------------------------------------------------------ #

    def _screen_chat(self, line: str) -> str:
        """Send a line through the thought pipeline."""
        from ultraquant.interpreter.thoughts import run_pipeline

        session = self.ensure_session()
        response, trace = run_pipeline(line, session)
        self._last_trace = trace
        return response

    def _screen_learn(self, line: str) -> str:
        """Gap survey and answering."""
        from ultraquant.interpreter.learning import LearningSession

        session = self.ensure_session()
        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        if verb in ("find", "survey"):
            self.learner = LearningSession(session)
            questions = self.learner.survey()
            if not questions:
                return "nothing to ask - the model has no gaps it can see"
            out = [f"{len(questions)} question(s) worth asking:"]
            for question in questions[:5]:
                out.append(f"  [{question.id}] {question.kind:<16} "
                           f"score {question.score:.2f}")
            out.append("")
            out.append(self._question_text())
            return "\n".join(out)
        if self.learner is None:
            return "run 'find' first"
        if verb == "research":
            reports = self.learner.research()
            if not reports:
                return ("nothing researchable is pending - research covers "
                        "unknown terms and topic categories")
            lines = [f"  {r['subject']:<24} {r['outcome']:<9} {r['detail']}"
                     for r in reports]
            return "\n".join(lines) + "\n\n" + self._question_text()
        if verb == "panel":
            return self._learn_panel(session, rest.split())
        if verb == "skip":
            question = self.learner.next_question()
            if question is None:
                return "nothing queued"
            self.learner.skip(question.id)
            return "skipped\n" + self._question_text()
        question = self.learner.next_question()
        if question is None:
            return "nothing queued - run 'find' again"
        answer = self.learner.answer(question, line)
        mark = "learned" if answer.accepted else "not applied"
        return f"{mark}: {answer.detail}\n\n" + self._question_text()

    def _learn_panel(self, session, models: list[str]) -> str:
        """Answer the queued question with the LLMLS panel.

        Parity with the chat CLI's ``:learn panel`` and the GUI's 'Ask the
        panel' button. A corroborated answer is *offered* - printed as the
        exact command that would apply it - never applied here. The panel is a
        source under interrogation, not an oracle.
        """
        from ultraquant.interpreter.llmls import LMStudioUnavailable, TeacherPanel

        question = self.learner.next_question()
        if question is None:
            return "nothing queued - run 'find' again"
        if question.expects == "glyph":
            return ("this question wants a glyph; a text panel cannot supply "
                    "pixels - answer with five rows instead")
        if not models:
            return ("usage: panel <model> [<model>...]  "
                    "(see the panel screen for the catalogue)")
        try:
            panel = TeacherPanel(models)
            result = panel.teach(
                question.prompt, session.stash,
                usages=question.context.get("usages"))
        except LMStudioUnavailable as exc:
            return f"LM Studio unavailable: {exc}"
        consensus = result["consensus"]
        out = [panel.independence_report(), consensus.as_text(),
               f"{result['filed']} claim(s) quarantined"]
        if consensus.corroborated:
            agreed = max(consensus.split.items(), key=lambda kv: len(kv[1]))[0]
            out.append(f"-> {consensus.voices} independent voices agree. To "
                       f"apply it, type:  {agreed}")
        else:
            out.append("-> not corroborated across independent voices; the "
                       "question stays open, which is the correct outcome")
        return "\n".join(out)

    def _question_text(self) -> str:
        """Render the queued question, glyph and all."""
        if self.learner is None:
            return ""
        question = self.learner.next_question()
        if question is None:
            return "no more questions"
        out = [self._paint(f"[{question.kind}]", "1;33") + " " + question.prompt]
        if question.options:
            out.append("  options: " + " / ".join(question.options))
        if question.expects == "glyph":
            out.append("  answer with five rows of five, using # and .")
        return "\n".join(out)

    def _screen_compute(self, line: str) -> str:
        """Tier detection and budgets."""
        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        if verb in ("detect", "", "show"):
            from ultraquant.native.dispatch import QUANTUM_TIERS

            out = ["quantum tiers:"]
            for tier in QUANTUM_TIERS:
                out.append(f"  {tier}")
            out.append("")
            out.append("status: " + self.status())
            from ultraquant.storage.ram import available_ram, total_ram

            out.append(f"system memory: {available_ram() / 2**30:.1f} GiB free "
                       f"of {total_ram() / 2**30:.1f} GiB")
            out.append(f"cpu threads available: {os.cpu_count()}")
            import json as _json

            for where in (self.home / "dispatch.json",
                          self.home / "forged" / "dispatch.json"):
                if not where.exists():
                    continue
                try:
                    payload = _json.loads(where.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - corrupt file is not a crash
                    continue
                out.append(f"learned dispatch: "
                           f"{len(payload.get('records', []))} timings")
                for kind, verdict in sorted(
                        payload.get("verdicts", {}).items()):
                    out.append(f"  {kind}: brain={verdict.get('chosen')} "
                               f"decide_us={verdict.get('decide_us')}")
            return "\n".join(out)
        if verb == "ram":
            session = self.ensure_session()
            megabytes = int(rest)
            session.cache.set_budget(megabytes * 1024 * 1024)
            return f"shard cache budget set to {megabytes} MB"
        return "commands: detect | ram <MB>"

    def _screen_forge(self, line: str) -> str:
        """Build a library."""
        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        if verb in ("build", ""):
            from ultraquant.forge.corpus import (
                BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
            )
            from ultraquant.forge.forge import ModelForge

            per_class = int(rest) if rest.strip().isdigit() else 24
            forge = ModelForge(self.home, seed=0, hidden=16, tier="auto")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                report = forge.build(
                    build_corpora(builtin_taxonomy(), n_per_class=per_class,
                                  seed=1, keywords=BUILTIN_KEYWORDS),
                    epochs=25,
                )
            self.session = None  # reopen so the new library is visible
            noise = buffer.getvalue().strip()
            return ((noise + "\n") if noise else "") + (
                report.summary()
                + f"\nlibrary: {len(forge.library_parts())} part(s)"
            )
        return "commands: build [examples-per-class]"

    def _screen_library(self, line: str) -> str:
        """The catalog."""
        session = self.ensure_session()
        verb = (line.split() or [""])[0].lower()
        if verb == "resident":
            resident = session.cache.stats()["resident"]
            return "resident: " + (", ".join(resident) or "(nothing)")
        if verb == "parts":
            libraries = session.vault.libraries()
            return "\n".join(f"  {p}" for p in libraries) or "no packed libraries"
        rows = [f"  {'shard':<28}{'category':<14}{'where':<10}{'bytes':>9}{'used':>6}"]
        for entry in session.vault.catalog()[:40]:
            rows.append(
                f"  {entry['shard_id']:<28}{entry['category']:<14}"
                f"{entry['location']:<10}{entry['nbytes']:>9}"
                f"{entry['access_count']:>6}"
            )
        if len(rows) == 1:
            return "the library is empty - forge one with ':forge build'"
        return "\n".join(rows)

    def _screen_storage(self, line: str) -> str:
        """Backends and locations."""
        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        if verb in ("show", ""):
            session = self.ensure_session()
            return (f"home    : {self.home}\n"
                    f"vault   : {session.vault.root}\n"
                    f"backend : {session.storage or 'local files'}")
        if verb == "use":
            from ultraquant.storage import open_storage

            storage = open_storage(rest)
            return f"opened {rest}: {storage}"
        if verb in ("library", "forge"):
            self.home = Path(rest)
            self.session = None
            return f"{verb} location set to {rest}"
        return "commands: show | use <uri> | library <dir> | forge <dir>"

    def _screen_stash(self, line: str) -> str:
        """Web-claim triage."""
        session = self.ensure_session()
        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        if verb in ("list", ""):
            entries = session.stash.entries()
            if not entries:
                return "the stash is empty - nothing has been fetched"
            return "\n".join(
                f"  [{e['id']}] {e['status']:<10}{e['classification']:<14}"
                f"{str(e['claim'])[:60]}"
                for e in entries[:30]
            )
        if verb in ("promote", "reject"):
            entry_id = int(rest)
            if verb == "promote":
                session.stash.promote(entry_id)
                return f"entry {entry_id} promoted to fact"
            session.stash.reject(entry_id)
            return f"entry {entry_id} rejected"
        return "commands: list | promote <id> | reject <id>"

    def _screen_panel(self, line: str) -> str:
        """The LLMLS teacher panel, at parity with the chat CLI's ':panel'.

        Every surface must reach every capability - the chat CLI gaining
        learning mode before the TUI did was recorded as a gap, and this
        avoids repeating it.
        """
        from ultraquant.interpreter.llmls import (
            LMStudioUnavailable, TeacherPanel, catalogue, independent_groups,
        )

        verb, _, rest = line.partition(" ")
        verb = verb.lower()
        try:
            if verb in ("list", ""):
                chat = [c for c in catalogue() if c.is_chat]
                if not chat:
                    return "LM Studio has no chat models"
                groups = independent_groups(chat)
                out = [f"{len(chat)} chat model(s) in {len(groups)} "
                       f"independent voice(s):"]
                for index, group in enumerate(groups, 1):
                    out.append(f"  voice {index} (arch={group[0].arch or '?'}, "
                               f"publisher={group[0].publisher or '?'}):")
                    for card in group:
                        out.append(f"    {'*' if card.loaded else ' '} {card.id}")
                out.append("  * = loaded. Voice counts are a LOWER BOUND on "
                           "correlation.")
                return "\n".join(out)
            if verb == "ask":
                models, _, question = rest.partition("?")
                names = models.split()
                if not names or not question.strip():
                    return "usage: ask <model> [<model>...] ? <question>"
                panel = TeacherPanel(names)
                consensus = panel.ask(question.strip())
                return (panel.independence_report() + "\n"
                        + consensus.as_text())
        except LMStudioUnavailable as exc:
            return f"LM Studio unavailable: {exc}"
        return "commands: list | ask <model>... ? <question>"

    def _screen_bench(self, line: str) -> str:
        """Measurements."""
        verb = (line.split() or ["run"])[0].lower()
        buffer = io.StringIO()
        if verb == "paging":
            from ultraquant.shards import scale_demo

            with redirect_stdout(buffer):
                scale_demo.main([])
        else:
            from ultraquant import bench

            with redirect_stdout(buffer):
                bench.main([])
        return buffer.getvalue().strip() or "(no output)"

    # ------------------------------------------------------------------ #
    # loop
    # ------------------------------------------------------------------ #

    def run(self, lines: Any | None = None) -> int:
        """Run the interface.

        Args:
            lines: Optional iterable of input lines, so a session can be
                scripted or replayed. Reads stdin when omitted.

        Returns:
            Process exit code.
        """
        self.write(self._paint("UltraQuant", "1;36") + f"  home={self.home}")
        self.write(self.status())
        self.write(self.help_text())
        self.write()
        self.write(self._screen_enter(self.screen))

        source = iter(lines) if lines is not None else None
        while self.running:
            self.write()
            self.write(self.banner())
            try:
                if source is None:
                    self.stream.write(f"{self.screen}> ")
                    self.stream.flush()
                    line = sys.stdin.readline()
                    if not line:
                        break
                else:
                    line = next(source)
                    self.write(f"{self.screen}> {line}")
            except (StopIteration, EOFError):
                break
            except KeyboardInterrupt:
                self.write("\ninterrupted - ':quit' to leave")
                continue
            out = self.handle(line)
            if out:
                self.write(out)
        if self.session is not None:
            try:
                self.session.save()
            except Exception:  # noqa: BLE001 - never fail on the way out
                pass
        return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m ultraquant.tui``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    home = None
    if argv and not argv[0].startswith("-"):
        home = argv[0]
    return UltraQuantTUI(home=home).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
