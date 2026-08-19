"""The terminal UI, and the portability it exists to serve.

A model library of this size lives on headless machines reached over SSH, where
Tkinter and a display are both absent. These tests drive the same eight surfaces
the desktop app offers, with no terminal attached at all — which is possible
because every screen is a function from a command to text.

They also cover the two portability traps that have already bitten this project
once each: non-ASCII output against a cp1252 console, and shared-library names
hardcoded to one platform's extension.
"""

from __future__ import annotations

import ast
import io
import os
import pathlib
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.tui import SCREENS, UltraQuantTUI


class TUIBasicsTests(unittest.TestCase):
    """Dispatch, navigation and help."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_tui_"))
        # Isolate the settings file: without this, every TUI test
        # that saves writes the developer's real settings - the full
        # suite once left a deleted temp dir as the standing home.
        self._old_config = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "settings.json")
        self.out = io.StringIO()
        self.tui = UltraQuantTUI(home=self.dir, stream=self.out, color=False)

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._old_config
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_every_screen_is_reachable_and_describes_itself(self) -> None:
        for name, blurb in SCREENS:
            with self.subTest(screen=name):
                text = self.tui.handle(f":{name}")
                self.assertEqual(self.tui.screen, name)
                self.assertIn(blurb, text)

    def test_every_screen_has_a_handler(self) -> None:
        """A screen in the menu with no handler would silently fall to chat."""
        for name, _blurb in SCREENS:
            with self.subTest(screen=name):
                self.assertTrue(hasattr(self.tui, f"_screen_{name}"))

    def test_help_lists_every_screen(self) -> None:
        text = self.tui.help_text()
        for name, _blurb in SCREENS:
            self.assertIn(f":{name}", text)

    def test_an_unknown_command_is_not_fatal(self) -> None:
        self.assertIn("unknown command", self.tui.handle(":nonsense"))
        self.assertTrue(self.tui.running)

    def test_quit_stops_the_loop(self) -> None:
        self.tui.handle(":quit")
        self.assertFalse(self.tui.running)

    def test_a_failing_command_reports_instead_of_exiting(self) -> None:
        """One bad input must not end the session."""
        self.tui.screen = "compute"
        text = self.tui.handle("ram not-a-number")
        self.assertIn("error", text.lower())
        self.assertTrue(self.tui.running)

    def test_a_screen_command_can_be_given_inline(self) -> None:
        self.tui.handle(":forge")
        text = self.tui.handle(":library list")
        self.assertEqual(self.tui.screen, "library")
        self.assertIsInstance(text, str)

    def test_home_can_be_repointed_and_drops_the_session(self) -> None:
        self.tui.ensure_session()
        self.assertIsNotNone(self.tui.session)
        other = self.dir / "elsewhere"
        self.tui.handle(f":home {other}")
        self.assertEqual(self.tui.home, other)
        self.assertIsNone(self.tui.session, "the old session must not leak across")

    def test_status_reports_without_a_session(self) -> None:
        self.assertTrue(self.tui.status())


class TUISessionTests(unittest.TestCase):
    """The surfaces that need a real library behind them."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_tuisession_"))
        # Isolate the settings file: without this, every TUI test
        # that saves writes the developer's real settings - the full
        # suite once left a deleted temp dir as the standing home.
        self._old_config = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "settings.json")
        self.out = io.StringIO()
        self.tui = UltraQuantTUI(home=self.dir, stream=self.out, color=False)

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._old_config
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_scripted_session_runs_end_to_end(self) -> None:
        """The whole point: no terminal, no display, still works."""
        self.tui.run([":forge build 20", ":library", ":storage show", ":quit"])
        text = self.out.getvalue()
        self.assertIn("experts", text, "the forge should report what it built")
        self.assertIn("mean accuracy", text)
        self.assertIn("expert:", text, "the catalog should list the forged experts")

    def test_a_glyph_typed_into_chat_reaches_the_library(self) -> None:
        self.tui.handle(":forge build 20")
        self.tui.handle(":chat")
        from ultraquant.pattern.recognition import PATTERNS

        response = self.tui.handle("\n".join(PATTERNS["square"]))
        self.assertIn("'square'", response)

    def test_the_trace_is_available_after_a_turn(self) -> None:
        self.tui.handle(":forge build 20")
        self.tui.handle(":chat")
        self.tui.handle("hello there")
        trace = self.tui.handle(":trace")
        for thought in ("Perceive", "Route", "Reason", "Respond"):
            self.assertIn(thought, trace)

    def test_library_reports_an_empty_catalog_plainly(self) -> None:
        self.assertIn("empty", self.tui.handle(":library"))

    def test_learn_requires_a_survey_first(self) -> None:
        self.tui.handle(":learn")
        self.assertIn("find", self.tui.handle("some answer"))

    def test_compute_reports_tiers_and_memory(self) -> None:
        text = self.tui.handle(":compute detect")
        self.assertIn("cpu threads", text)
        self.assertIn("system memory", text)


class ConsoleSafetyTests(unittest.TestCase):
    """Output has to survive the console it lands on."""

    def test_no_printed_string_is_non_ascii(self) -> None:
        """Docstrings may use typography; anything printed may not.

        A Windows console still defaults to cp1252, where an em-dash becomes a
        replacement character mid-sentence. This has already happened twice in
        this project, so it is asserted rather than remembered.
        """
        source = pathlib.Path("ultraquant/tui.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        offenders = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in docstrings
            and any(ord(ch) > 127 for ch in node.value)
        ]
        self.assertEqual(offenders, [])

    def test_every_screen_encodes_as_cp1252(self) -> None:
        tui = UltraQuantTUI(home="uq_probe", stream=io.StringIO(), color=False)
        text = tui.help_text()
        for name, _blurb in SCREENS:
            tui.screen = name
            text += tui._screen_enter(name) + tui.banner()
        text.encode("cp1252")  # raises if anything is unencodable

    def test_output_survives_a_stream_that_cannot_encode_it(self) -> None:
        """Model output is not ours to sanitise; it must not crash the UI."""

        class AsciiOnly(io.StringIO):
            encoding = "ascii"

            def write(self, text):
                text.encode("ascii")  # raises like a real console would
                return super().write(text)

        stream = AsciiOnly()
        tui = UltraQuantTUI(home="uq_probe", stream=stream, color=False)
        tui.write("safe")
        tui.write("café — naïve")  # would raise unhandled
        self.assertIn("safe", stream.getvalue())

    def test_colour_is_off_when_the_stream_is_not_a_terminal(self) -> None:
        tui = UltraQuantTUI(home="uq_probe", stream=io.StringIO())
        self.assertFalse(tui.color, "a pipe or log file must not get escapes")
        self.assertNotIn("\033", tui.banner())


class PortabilityTests(unittest.TestCase):
    """The native tiers must be loadable on the platform they were built for."""

    def test_library_names_follow_the_platform(self) -> None:
        """Hardcoding '.dll' made the accelerators unloadable off Windows.

        The failure was silent — the pure-Python tier took over and the only
        symptom was that everything ran slower.
        """
        from ultraquant.native.accel import _library_name

        import ultraquant.native.accel as accel

        original_name, original_platform = accel.os.name, accel.sys.platform
        try:
            accel.os.name = "nt"
            self.assertEqual(_library_name("uq"), "uq.dll")
            accel.os.name = "posix"
            accel.sys.platform = "linux"
            self.assertEqual(_library_name("uq"), "libuq.so")
            accel.sys.platform = "darwin"
            self.assertEqual(_library_name("uq"), "libuq.dylib")
        finally:
            accel.os.name, accel.sys.platform = original_name, original_platform

    def test_the_forge_uses_the_same_resolution(self) -> None:
        """Both accelerator families, or half the tiers stay Windows-only."""
        from ultraquant.forge import trainer

        for path in (trainer._CPU_DLL, trainer._GPU_DLL):
            self.assertTrue(
                path.name.endswith((".dll", ".so", ".dylib")), path.name
            )

    def test_build_hints_name_the_right_script(self) -> None:
        import ultraquant.native.accel as accel

        original = accel.os.name
        try:
            accel.os.name = "nt"
            self.assertIn("build.ps1", accel._build_hint("cpu"))
            accel.os.name = "posix"
            self.assertIn("build.sh", accel._build_hint("cpu"))
        finally:
            accel.os.name = original

    def test_the_posix_launcher_and_build_script_exist(self) -> None:
        for name in ("ultraquant.sh", "native/build.sh"):
            with self.subTest(script=name):
                path = pathlib.Path(name)
                self.assertTrue(path.is_file(), f"{name} is missing")
                text = path.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("#!"), "needs a shebang")
                self.assertNotIn("\r\n", text, "CRLF breaks a shell script on Linux")

    def test_the_launcher_offers_the_tui_when_tk_is_absent(self) -> None:
        """Headless Linux is the case the TUI exists for."""
        text = pathlib.Path("ultraquant.sh").read_text(encoding="utf-8")
        self.assertIn("ultraquant.tui", text)
        self.assertIn("tkinter", text)


if __name__ == "__main__":
    unittest.main()
