"""The self-learning loop, driven through the chat CLI.

The GUI and TUI had learning mode; the chat CLI did not until the loop was
actually driven end to end here and the gap showed. These tests hold the whole
cycle in place at the chat surface:

    doubt      -> :learn -> shaky question -> answer -> confidence rises
    unfamiliar -> :recognize x N -> proposed-concept -> name it -> recognised

Everything goes through ``ChatCLI.handle`` — the same entry point the REPL,
``--script`` and ``--once`` forms use — so what is tested is what a user at the
prompt actually gets.
"""

from __future__ import annotations

import io
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.forge.languages import seed_knowledge
from ultraquant.interpreter.chat import ChatCLI
from ultraquant.interpreter.thoughts import build_session


class ChatSelfLearnTests(unittest.TestCase):
    """The loop at the chat surface."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatlearn_"))
        self.session = build_session(self.dir, seed=0)
        seed_knowledge(self.session)
        self.out = io.StringIO()
        self.cli = ChatCLI(self.session, out=self.out)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _say(self, line: str, rows: list[str] | None = None) -> str:
        start = self.out.tell()
        self.cli.handle(line, more=iter(rows) if rows else None)
        return self.out.getvalue()[start:]

    def _skip_to(self, kind: str, limit: int = 20) -> bool:
        for _ in range(limit):
            question = self.cli.learning.next_question()
            if question is None:
                return False
            if question.kind == kind:
                return True
            self._say(":learn skip")
        return False

    # -- phase A: doubt -----------------------------------------------------

    def test_confirming_a_shaky_fact_through_chat_raises_confidence(self) -> None:
        before = self._say(":facts densest")
        self.assertIn("0.40", before)

        survey = self._say(":learn")
        self.assertIn("question(s) worth asking", survey)
        self.assertIn("shaky", survey)

        self.assertTrue(self._skip_to("shaky"))
        applied = self._say(":learn answer yes")
        self.assertIn("learned", applied)

        after = self._say(":facts densest")
        self.assertIn("0.90", after)

    def test_learn_answer_without_a_survey_is_survivable(self) -> None:
        self.assertIn("Run ':learn' first", self._say(":learn answer yes"))

    def test_the_help_advertises_the_loop(self) -> None:
        self.assertIn(":learn", self._say(":help"))

    # -- phase B: unfamiliar -> concept -> known ----------------------------

    def _show_families(self) -> list[str]:
        """Feed 28 unfamiliar patterns; return one prototype's rows."""
        from ultraquant.experiments.discovery_gate import STROKES

        def rows_of(lit):
            return ["".join("#" if r * 5 + c in lit else "." for c in range(5))
                    for r in range(5)]

        rng = random.Random(0)
        first = set(STROKES["slash"]) | set(STROKES["top"])
        second = set(STROKES["left"]) | set(STROKES["bottom"])
        for family in (first, second):
            for _ in range(14):
                lit = set(family)
                for position in rng.sample(range(25), 3):
                    lit.symmetric_difference_update({position})
                self._say(":recognize", rows=rows_of(lit))
        return rows_of(first)

    def test_the_full_cycle_unfamiliar_to_recognised(self) -> None:
        prototype = self._show_families()

        survey = self._say(":learn")
        self.assertIn("proposed-concept", survey)

        self.assertTrue(self._skip_to("proposed-concept"))
        named = self._say(":learn answer diagonals")
        self.assertIn("named the group 'diagonals'", named)

        reading = self._say(":recognize", rows=prototype)
        self.assertIn("'diagonals'", reading)
        self.assertNotIn("do not recognise", reading)

    def test_a_glyph_expecting_question_reads_five_rows(self) -> None:
        """The other answer mode: untrained categories ask for a demonstration."""
        self._say("show me some geometry please")
        self._say(":learn")
        if not self._skip_to("untrained-category"):
            self.skipTest("no untrained-category question surfaced")
        applied = self._say(
            ":learn answer",
            rows=["#####", "#...#", "#...#", "#...#", "#####"],
        )
        self.assertIn("learned", applied)


class RemedyTests(unittest.TestCase):
    """The question must ask for what would actually help."""

    def setUp(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        self.dir = Path(tempfile.mkdtemp(prefix="uq_remedy_"))
        self.session = build_session(self.dir, seed=0)
        self.learner = LearningSession(self.session)
        self.questions = {
            q.subject: q for q in self.learner.survey()
            if q.kind == "untrained-category"
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_topic_categories_ask_for_knowledge_not_glyphs(self) -> None:
        """A drawn glyph teaches 'arithmetic' nothing, and the first
        end-to-end drive of the loop showed exactly that absurd request."""
        for topic in ("arithmetic", "language", "world"):
            with self.subTest(topic=topic):
                question = self.questions[topic]
                self.assertEqual(question.expects, "text")
                self.assertIn("Tell me something", question.prompt)
                self.assertNotIn("five rows", question.prompt)

    def test_the_glyph_fallback_still_asks_for_a_demonstration(self) -> None:
        question = self.questions["geometry"]
        self.assertEqual(question.expects, "glyph")
        self.assertIn("five rows", question.prompt)

    def test_a_topic_answer_stores_facts_and_routes_to_the_category(self) -> None:
        question = self.questions["arithmetic"]
        answer = self.learner.answer(
            question, "arithmetic is the study of numbers and operations"
        )
        self.assertTrue(answer.accepted, answer.detail)
        fact = self.session.memory.recall_fact("arithmetic")
        self.assertIn("numbers", str(fact["value"]))
        ranked = dict(self.session.router.route("the study of numbers"))
        self.assertIn("arithmetic", ranked)

    def test_a_topic_answer_without_a_statement_is_declined(self) -> None:
        question = self.questions["world"]
        answer = self.learner.answer(question, "hmm not sure")
        self.assertFalse(answer.accepted)


class ResearchTests(unittest.TestCase):
    """The web is the first resort; the human is the second."""

    PAGE = """<html><head><title>Pixels</title></head><body>
    <p>Pixels is the smallest addressable element of a raster image.</p>
    <p>Some people think pixels are beautiful.</p>
    </body></html>"""

    @classmethod
    def setUpClass(cls) -> None:
        import http.server
        import threading

        page = cls.PAGE.encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.source = f"http://127.0.0.1:{cls.server.server_address[1]}/{{term}}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        self.dir = Path(tempfile.mkdtemp(prefix="uq_research_"))
        self.session = build_session(self.dir, seed=0, online=True)
        # Make 'pixels' a repeated unknown term so the survey asks about it.
        for _ in range(4):
            self.session.memory.remember_episode(
                "interaction", {"text": "tell me about pixels"}, tags=["chat"]
            )
        self.learner = LearningSession(self.session)
        self.learner.survey()

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _pixels_pending(self) -> bool:
        return any(q.subject == "pixels" for q in self.learner.pending
                   if q.kind == "unknown-term")

    def test_research_resolves_an_unknown_term_through_the_stash(self) -> None:
        self.assertTrue(self._pixels_pending())
        reports = self.learner.research(sources=[self.source])
        by_subject = {r["subject"]: r for r in reports}
        self.assertEqual(by_subject["pixels"]["outcome"], "resolved")

        fact = self.session.memory.recall_fact("pixels")
        self.assertIn("raster", str(fact["value"]))
        self.assertFalse(self._pixels_pending(),
                         "a researched question must leave the queue")

    def test_found_is_still_not_believed(self) -> None:
        """The opinion sentence must stay quarantined, not become fact."""
        self.learner.research(sources=[self.source])
        opinions = [e for e in self.session.stash.entries()
                    if e["classification"] == "opinion"]
        self.assertTrue(opinions, "the opinion should be in the stash")
        for entry in opinions:
            self.assertNotEqual(entry["status"], "promoted")

    def test_research_respects_the_online_gate(self) -> None:
        self.session.web.online = False
        reports = self.learner.research(sources=[self.source])
        self.assertTrue(reports)
        self.assertEqual(reports[0]["outcome"], "offline")
        self.assertIsNone(self.session.memory.recall_fact("pixels"))

    def test_the_chat_surface_reaches_research(self) -> None:
        out = io.StringIO()
        cli = ChatCLI(self.session, out=out)
        cli.learning = self.learner
        cli.handle(":learn research")
        text = out.getvalue()
        self.assertIn("resolved", text)


class IntakeQualityTests(unittest.TestCase):
    """The web intake fixes, each tied to a live-measured failure."""

    JUNK_FIRST = """<html><head><title>Pixel</title></head><body>
    <p>For other uses, see Pixel (disambiguation).</p>
    <p>"Picture element" redirects here.</p>
    <p>For the phone, see Pixel (phone).</p>
    <p>See also: Voxel.</p>
    <p>Not to be confused with Pixels (film).</p>
    <p>This page was last edited by a volunteer contributor recently.</p>
    <p>Navigation menu appears on every page of the encyclopedia site.</p>
    <p>Jump to search from the main navigation content block here.</p>
    <p>A pixel is the smallest addressable element of a raster image.</p>
    </body></html>"""

    @classmethod
    def setUpClass(cls) -> None:
        import http.server
        import threading

        def serve(payload: bytes):
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802 - http.server API
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server

        page = cls.JUNK_FIRST.encode("utf-8")
        cls.servers = [serve(page), serve(page)]
        cls.sources = [
            f"http://127.0.0.1:{s.server_address[1]}/{{term}}"
            for s in cls.servers
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        for server in cls.servers:
            server.shutdown()

    def setUp(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        self.dir = Path(tempfile.mkdtemp(prefix="uq_intake_"))
        self.session = build_session(self.dir, seed=0, online=True)
        for _ in range(4):
            self.session.memory.remember_episode(
                "interaction", {"text": "tell me about pixels"}, tags=["chat"]
            )
        self.learner = LearningSession(self.session)
        self.learner.survey()

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_navigation_chrome_never_becomes_a_claim(self) -> None:
        """Hatnotes and cross-references are pointers, not claims."""
        self.learner.research(sources=self.sources[:1])
        claims = [e["claim"].lower() for e in self.session.stash.entries()]
        self.assertTrue(claims)
        for claim in claims:
            self.assertNotIn("redirects here", claim)
            self.assertNotIn("disambiguation", claim)
            self.assertFalse(claim.startswith("for "))

    def test_the_definition_survives_a_junk_front_load(self) -> None:
        """Live failure: eight lines of chrome ate the claim budget and the
        sentence defining 'pixel' never made the cut."""
        reports = self.learner.research(sources=self.sources[:1])
        by_subject = {r["subject"]: r for r in reports}
        self.assertEqual(by_subject["pixels"]["outcome"], "resolved")

    def test_two_sources_corroborate_and_raise_confidence(self) -> None:
        self.learner.research(sources=self.sources)
        fact = self.session.memory.recall_fact("pixel")
        self.assertIsNotNone(fact)
        self.assertGreaterEqual(fact["confidence"], 0.8,
                                "two netlocs agreeing should corroborate")

    def test_a_qualified_lead_sentence_beats_a_worse_match(self) -> None:
        """Live failure: "In communications..., code is a system of rules" was
        excluded by a length cap, so "An early example of a code is language"
        won and 'what is code?' answered that code IS language."""
        import http.server
        import threading

        page = ("<html><head><title>Code</title></head><body>"
                "<p>An early example of a code is language, which enables "
                "communication between people who share it.</p>"
                "<p>In communications and information processing, code is a "
                "system of rules to convert information into another form.</p>"
                "</body></html>").encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for _ in range(4):
                self.session.memory.remember_episode(
                    "interaction", {"text": "explain the code please"},
                    tags=["chat"],
                )
            from ultraquant.interpreter.learning import LearningSession

            learner = LearningSession(self.session)
            learner.survey()
            source = f"http://127.0.0.1:{server.server_address[1]}/{{term}}"
            learner.research(sources=[source])
            fact = self.session.memory.recall_fact("code")
            self.assertIsNotNone(fact)
            self.assertIn("system of rules", str(fact["value"]),
                          "the defining sentence must win, not the example")
        finally:
            server.shutdown()

    def test_the_bare_term_recalls_after_resolution(self) -> None:
        """Live failure: a question about 'code' was closed with the fact
        stored only under 'early example of a code' - unreachable by the very
        question it answered."""
        self.learner.research(sources=self.sources[:1])
        self.assertIsNotNone(self.session.memory.recall_fact("pixels"))


if __name__ == "__main__":
    unittest.main()
