"""End-to-end tests for the Chat/Interpreter: pipeline, learning, and the CLI."""

from __future__ import annotations

import http.server
import io
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from ultraquant.interpreter.chat import ChatCLI, main
from ultraquant.interpreter.selflearn import SelfLearner
from ultraquant.interpreter.thoughts import (
    PIPELINE,
    build_session,
    run_pipeline,
)
from ultraquant.pattern.recognition import PATTERNS

CLAIM_PAGE = """<html><head><title>Facts</title></head><body>
<p>The orbital period is 687 days.</p>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve one fixed page of claims."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = CLAIM_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Silence the server."""


def _serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    """Start a throwaway HTTP server on a free port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class PipelineTests(unittest.TestCase):
    """The predefined thought pipeline behaves as specified."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_interp_"))
        self.session = build_session(self.dir, budget_bytes=256 * 1024, seed=3)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_trace_follows_the_predefined_order(self) -> None:
        _response, trace = run_pipeline("hello there", self.session)
        self.assertEqual(
            [step["thought"] for step in trace],
            [thought.name for thought in PIPELINE],
        )

    def test_fact_round_trip(self) -> None:
        run_pipeline("the sky color is blue", self.session)
        response, _trace = run_pipeline("what is the sky color?", self.session)
        self.assertIn("blue", response.lower())

    def test_code_intent_computes(self) -> None:
        response, _trace = run_pipeline("calc: sum([x*x for x in range(1, 11)])", self.session)
        self.assertIn("385", response)

    def test_glyph_recognition_and_signature(self) -> None:
        """A glyph reaches a trained expert and its reading is remembered."""
        rows = PATTERNS["plus"]
        SelfLearner(self.session).teach_glyph(
            "geometry", ["plus", "square"], rows, "plus"
        )
        response, _trace = run_pipeline("\n".join(rows), self.session)
        self.assertIn("reads as", response)
        self.assertGreater(self.session.memory.stats()["signatures"], 0)

    def test_an_empty_library_declines_to_guess_at_a_glyph(self) -> None:
        """With nothing trained, an answer would be a coin toss dressed as fact.

        Measured before this was fixed: a fresh session scored **0/8** on the
        built-in glyphs while reporting confidences of 0.16-0.35, because it
        invented an expert on the spot and read out its random initialisation.
        """
        response, _trace = run_pipeline("\n".join(PATTERNS["plus"]), self.session)
        self.assertNotIn("reads as", response)
        self.assertIn("no pattern library", response)
        self.assertTrue(
            self.session.memory.recall_episodes(kind="unknown-pattern", limit=5),
            "an unreadable pattern should still become something to learn",
        )

    def test_routing_reports_paging(self) -> None:
        _response, trace = run_pipeline("what shape is this square glyph?", self.session)
        route_step = next(s for s in trace if s["thought"] == "Route")
        self.assertIn("would_load", route_step)

    def test_unknown_question_is_honest(self) -> None:
        response, _trace = run_pipeline("what is the capital of atlantis?", self.session)
        self.assertIn("don't hold", response.lower())


class WebIntakeTests(unittest.TestCase):
    """Web content is quarantined, and only corroboration promotes it."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_web_"))
        self.session = build_session(self.dir, budget_bytes=256 * 1024, seed=1, online=True)
        # Two servers on different ports give two distinct netlocs, which is the
        # local stand-in for two independent sources.
        self.server_a, self.port_a = _serve()
        self.server_b, self.port_b = _serve()

    def tearDown(self) -> None:
        for server in (self.server_a, self.server_b):
            server.shutdown()
            server.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_first_fetch_stages_without_storing_fact(self) -> None:
        response, _trace = run_pipeline(f"http://127.0.0.1:{self.port_a}/", self.session)
        self.assertIn("stashed", response.lower())
        self.assertIn("nothing has been stored as fact", response.lower())
        self.assertEqual(self.session.memory.stats()["semantic"], 0)
        self.assertGreater(self.session.stash.stats()["entries"], 0)

    def test_second_source_corroborates_and_promotes(self) -> None:
        run_pipeline(f"http://127.0.0.1:{self.port_a}/", self.session)
        response, _trace = run_pipeline(f"http://127.0.0.1:{self.port_b}/", self.session)
        self.assertIn("promoted to fact", response.lower())
        fact = self.session.memory.recall_fact("orbital period")
        self.assertIsNotNone(fact)
        self.assertIn("687", str(fact["value"]))
        self.assertAlmostEqual(fact["confidence"], 0.8)

    def test_offline_refuses_and_stores_nothing(self) -> None:
        self.session.web.set_online(False)
        response, _trace = run_pipeline(f"http://127.0.0.1:{self.port_a}/", self.session)
        self.assertIn("web access is off", response.lower())
        self.assertEqual(self.session.stash.stats()["entries"], 0)


class SelfLearningTests(unittest.TestCase):
    """Teaching accumulates and consolidation packs and snapshots."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_learn_"))
        self.session = build_session(self.dir, budget_bytes=256 * 1024, seed=5)
        self.learner = SelfLearner(self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_extract_facts(self) -> None:
        facts = self.learner.extract_facts("remember that the moon radius is 1737 km")
        self.assertEqual(facts, [("moon radius", "1737 km")])

    def test_teach_glyph_trains_expert(self) -> None:
        labels = ["plus", "square"]
        self.session.experts.ensure_expert("geometry", labels)
        report = self.learner.teach_glyph("geometry", labels, PATTERNS["plus"], "plus")
        self.assertGreaterEqual(report["accuracy"], 0.8)
        label, _confidence = self.session.experts.predict(
            "geometry",
            _features(PATTERNS["plus"]),
        )
        self.assertEqual(label, "plus")

    def test_consolidate_packs_and_snapshots(self) -> None:
        labels = ["plus", "square"]
        self.session.experts.ensure_expert("geometry", labels)
        # Two accesses make the shard "hot" enough to consolidate.
        for _ in range(3):
            self.session.experts.predict("geometry", _features(PATTERNS["plus"]))
            self.session.cache.invalidate("expert:geometry")

        report = self.learner.consolidate()
        self.assertGreaterEqual(report["packed"], 1)
        self.assertTrue(Path(report["library"]).exists())
        self.assertTrue(report["snapshot"].startswith("T-"))

        # The shard now lives inside the library and is still readable by chunk.
        entry = self.session.vault.entry("expert:geometry")
        self.assertEqual(entry["location"], "library")
        payload = self.session.vault.get("expert:geometry")
        self.assertIn("net", payload)

        restored = self.session.archive.restore(report["snapshot"])
        self.assertIn("vault", restored)


class ChatCLITests(unittest.TestCase):
    """The CLI runs scripted and one-shot without a terminal."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_cli_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_script_mode_exercises_commands(self) -> None:
        script = self.dir / "session.txt"
        script.write_text(
            "\n".join(
                [
                    ":help",
                    "the tower height is 324 metres",
                    "what is the tower height?",
                    ":facts tower",
                    ":code 6 * 7",
                    ":shards",
                    ":resident",
                    ":stash",
                    ":analyze",
                    ":budget 64",
                    ":consolidate",
                    ":quit",
                ]
            ),
            encoding="utf-8",
        )
        buffer = io.StringIO()
        session = build_session(self.dir / "home", budget_bytes=128 * 1024, seed=0)
        cli = ChatCLI(session, out=buffer)
        lines = script.read_text(encoding="utf-8").splitlines()
        stream = iter(lines)
        for line in stream:
            cli.handle(line, stream)
            if not cli.running:
                break
        output = buffer.getvalue()

        self.assertIn("UltraQuant Chat/Interpreter - commands", output)
        self.assertIn("Noted: tower height is 324 metres", output)
        self.assertIn("324 metres", output)
        self.assertIn("= 42", output)
        self.assertIn("budget", output)
        self.assertIn("Stash is empty", output)
        self.assertIn("Exiting", output.replace("exiting", "Exiting"))

    def test_once_mode_exits_zero(self) -> None:
        code = main(["--root", str(self.dir / "once"), "--once", "calc: 2 ** 10"])
        self.assertEqual(code, 0)

    def test_script_flag_exits_zero(self) -> None:
        script = self.dir / "s.txt"
        script.write_text("calc: 1 + 1\n:quit\n", encoding="utf-8")
        code = main(["--root", str(self.dir / "home2"), "--script", str(script)])
        self.assertEqual(code, 0)

    def test_output_survives_a_legacy_console(self) -> None:
        """Every CLI string must encode on a cp1252 console, not just in a StringIO.

        Regression guard: an arrow glyph in ':resident' once crashed the real
        Windows console with UnicodeEncodeError while every test still passed,
        because tests capture to a StringIO that accepts any character.
        """
        from ultraquant.interpreter.chat import HELP

        buffer = io.StringIO()
        session = build_session(self.dir / "ascii", seed=0)
        cli = ChatCLI(session, out=buffer)
        for command in (":help", ":mem", ":facts", ":shards", ":resident", ":stash", ":trace"):
            cli.handle(command)
        for text in (HELP, buffer.getvalue()):
            try:
                text.encode("cp1252")
            except UnicodeEncodeError as exc:
                self.fail(f"CLI output is not console-safe: {exc}")

    def test_unknown_command_is_survivable(self) -> None:
        buffer = io.StringIO()
        session = build_session(self.dir / "home3", seed=0)
        cli = ChatCLI(session, out=buffer)
        cli.handle(":nonsense")
        self.assertIn("Unknown command", buffer.getvalue())


def _features(rows: list[str]) -> list[float]:
    """Build the 30-dim feature vector for a glyph."""
    from ultraquant.pattern.recognition import render, row_means

    pixels = render(rows)
    return pixels + row_means(pixels)


if __name__ == "__main__":
    unittest.main()
