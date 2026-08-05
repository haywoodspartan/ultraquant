"""Tests for the code function, gated web access, and the contemporary stash."""

from __future__ import annotations

import http.server
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from ultraquant.interpreter.codefunc import CodeError, SafeCodeRunner
from ultraquant.interpreter.stash import ContemporaryStash, StashError
from ultraquant.interpreter.webaccess import WebAccess, WebDisabled
from ultraquant.memory.systematic import SystematicMemory

PAGE = """<html><head><title>Test Page</title></head><body>
<script>var hidden = "should not appear";</script>
<p>The tower height is 324 metres.</p>
<p>I think the tower is the most beautiful structure ever built.</p>
<p>The tower reportedly sways up to 9 centimetres in the wind.</p>
</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve one fixed HTML page."""

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Silence the server."""


class SafeCodeRunnerTests(unittest.TestCase):
    """The sandbox computes what it should and refuses what it must."""

    def setUp(self) -> None:
        self.runner = SafeCodeRunner(max_ops=20_000, timeout_s=5.0)

    def test_arithmetic_result(self) -> None:
        self.assertEqual(self.runner.run("2 + 3 * 4")["result"], 14)

    def test_function_definition_and_call(self) -> None:
        result = self.runner.run("def square(n):\n    return n * n\nsquare(7)")
        self.assertEqual(result["result"], 49)
        self.assertIn("square", result["defined"])

    def test_loop_accumulator(self) -> None:
        source = "total = 0\nfor i in range(1, 11):\n    total = total + i\ntotal"
        self.assertEqual(self.runner.run(source)["result"], 55)

    def test_math_whitelist(self) -> None:
        self.assertAlmostEqual(self.runner.run("math.sqrt(144)")["result"], 12.0)

    def test_fstring_and_comprehension(self) -> None:
        result = self.runner.run("xs = [x*x for x in range(4)]\nf'{xs}'")
        self.assertEqual(result["result"], "[0, 1, 4, 9]")

    def test_print_is_captured(self) -> None:
        result = self.runner.run("print('hello', 'world')")
        self.assertEqual(result["stdout"], "hello world\n")

    def test_rejects_import(self) -> None:
        with self.assertRaises(CodeError):
            self.runner.run("import os")

    def test_rejects_open(self) -> None:
        with self.assertRaises(CodeError):
            self.runner.run("open('secrets.txt')")

    def test_rejects_dunder_escape(self) -> None:
        for source in (
            "().__class__",
            "(1).__class__.__bases__",
            "x = 1\nx.__dict__",
            "__import__('os')",
        ):
            with self.subTest(source=source), self.assertRaises(CodeError):
                self.runner.run(source)

    def test_rejects_non_math_attribute(self) -> None:
        with self.assertRaises(CodeError):
            self.runner.run("'abc'.upper()")

    def test_rejects_unavailable_math_function(self) -> None:
        with self.assertRaises(CodeError):
            self.runner.run("math.factorial(5)")

    def test_rejects_exec_and_eval(self) -> None:
        for source in ("exec('x=1')", "eval('1+1')", "getattr(math, 'pi')"):
            with self.subTest(source=source), self.assertRaises(CodeError):
                self.runner.run(source)

    def test_operation_budget_stops_infinite_loop(self) -> None:
        runner = SafeCodeRunner(max_ops=5_000, timeout_s=10.0)
        with self.assertRaises(CodeError):
            runner.run("while True:\n    pass")

    def test_result_shape(self) -> None:
        result = self.runner.run("a = 1\na")
        self.assertEqual(sorted(result), ["defined", "result", "stdout"])


class WebAccessTests(unittest.TestCase):
    """Access is gated, scheme-checked, and reduces HTML to text."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_offline_refuses(self) -> None:
        with self.assertRaises(WebDisabled):
            WebAccess(online=False).fetch("http://127.0.0.1/")

    def test_bad_scheme(self) -> None:
        with self.assertRaises(ValueError):
            WebAccess(online=True).fetch("ftp://example.com/file")

    def test_fetch_extracts_title_and_strips_script(self) -> None:
        page = WebAccess(online=True).fetch(f"http://127.0.0.1:{self.port}/")
        self.assertEqual(page["title"], "Test Page")
        self.assertIn("324 metres", page["text"])
        self.assertNotIn("should not appear", page["text"])

    def test_ssl_context_verifies_and_excludes_expired_roots(self) -> None:
        """Trust anchors must be current, and verification must stay on.

        Regression guard: this machine's Windows store held an expired copy of
        Let's Encrypt's ISRG Root X2. OpenSSL matches a trust anchor by subject
        name, so it chained to the stale copy and every Wikipedia fetch died
        with "certificate has expired" even though the server's chain was valid.
        """
        import datetime
        import ssl

        from ultraquant.interpreter.webaccess import build_ssl_context

        context = build_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

        cas = context.get_ca_certs()
        self.assertGreater(len(cas), 10, "trust store must not be gutted")

        now = datetime.datetime.now(datetime.timezone.utc)
        expired = []
        for ca in cas:
            raw = ca.get("notAfter")
            if not raw:
                continue
            try:
                when = datetime.datetime.strptime(
                    raw, "%b %d %H:%M:%S %Y %Z"
                ).replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
            if when <= now:
                subject = dict(x[0] for x in ca.get("subject", ()))
                expired.append(subject.get("commonName", "?"))
        self.assertEqual(expired, [], f"expired trust anchors kept: {expired}")

    def test_ssl_context_is_cached(self) -> None:
        from ultraquant.interpreter.webaccess import build_ssl_context

        self.assertIs(build_ssl_context(), build_ssl_context())

    def test_ssl_context_honours_a_custom_ca_bundle(self) -> None:
        """A corporate TLS-inspecting proxy is handled by trusting its root."""
        import ssl

        from ultraquant.interpreter.webaccess import build_ssl_context

        bundle = Path(tempfile.mkdtemp(prefix="uq_ca_")) / "corp.pem"
        try:
            bundle.write_text(_SELF_SIGNED_PEM, encoding="utf-8")
            context = build_ssl_context(str(bundle))
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            subjects = [
                dict(x[0] for x in ca.get("subject", ())).get("commonName")
                for ca in context.get_ca_certs()
            ]
            self.assertIn("uq-test-ca", subjects)
        finally:
            shutil.rmtree(bundle.parent, ignore_errors=True)

    def test_web_access_has_no_memory_writer(self) -> None:
        # The web layer must not be able to write knowledge directly.
        self.assertFalse(
            [name for name in dir(WebAccess) if "learn" in name or "remember" in name]
        )


class ContemporaryStashTests(unittest.TestCase):
    """Web claims are quarantined, classified, and gated before becoming fact."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_stash_"))
        self.stash = ContemporaryStash(self.dir / "stash.json")
        self.memory = SystematicMemory(path=self.dir / "memory.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_page_splits_claims(self) -> None:
        ids = self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.assertGreaterEqual(len(ids), 3)

    def test_classification(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        classes = {
            "height": None, "opinion": None, "hedge": None,
        }
        for entry in self.stash.entries():
            if entry["claim"].startswith("The tower height"):
                classes["height"] = entry["classification"]
            elif entry["claim"].startswith("I think"):
                classes["opinion"] = entry["classification"]
            elif "reportedly" in entry["claim"]:
                classes["hedge"] = entry["classification"]
        self.assertEqual(classes["height"], "factual-claim")
        self.assertEqual(classes["opinion"], "opinion")
        self.assertEqual(classes["hedge"], "hedged")

    def test_second_source_corroborates(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        entry = self._factual()
        self.assertEqual(entry["status"], "staged")

        self.stash.add_page("http://b.test/y", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        entry = self._factual()
        self.assertEqual(entry["status"], "corroborated")
        self.assertEqual(len(entry["sources"]), 2)

    def test_contradiction_is_disputed(self) -> None:
        self.memory.remember_fact("tower height", "1000 metres", confidence=0.9)
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        self.assertEqual(self._factual()["status"], "disputed")

    def test_promote_corroborated_writes_fact(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.add_page("http://b.test/y", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        key = self.stash.promote(self._factual()["id"], self.memory)
        fact = self.memory.recall_fact(key)
        self.assertIsNotNone(fact)
        self.assertAlmostEqual(fact["confidence"], 0.8)
        self.assertEqual(self._factual()["status"], "promoted")
        self.assertTrue(self.memory.recall_episodes(kind="promotion"))

    def test_opinion_needs_force(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        opinion = next(e for e in self.stash.entries() if e["classification"] == "opinion")
        with self.assertRaises(StashError):
            self.stash.promote(opinion["id"], self.memory)
        self.assertIsInstance(self.stash.promote(opinion["id"], self.memory, force=True), str)

    def test_disputed_needs_force(self) -> None:
        self.memory.remember_fact("tower height", "1000 metres", confidence=0.9)
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        with self.assertRaises(StashError):
            self.stash.promote(self._factual()["id"], self.memory)

    def test_reject(self) -> None:
        ids = self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.reject(ids[0], "unreliable source")
        entry = self.stash.get(ids[0])
        self.assertEqual(entry["status"], "rejected")
        self.assertEqual(entry["notes"], "unreliable source")

    def test_persistence_round_trip(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        reopened = ContemporaryStash(self.dir / "stash.json")
        self.assertEqual(reopened.stats(), self.stash.stats())

    def test_nothing_reaches_memory_without_promotion(self) -> None:
        self.stash.add_page("http://a.test/x", "T", PAGE_TEXT)
        self.stash.analyze(self.memory)
        self.assertEqual(self.memory.stats()["semantic"], 0)

    def _factual(self) -> dict:
        """The height claim entry."""
        return next(e for e in self.stash.entries() if e["claim"].startswith("The tower height"))


#: A throwaway self-signed CA, used to check that a custom bundle is honoured
#: (the path a TLS-inspecting corporate proxy would take). Valid until 2046.
_SELF_SIGNED_PEM = """
-----BEGIN CERTIFICATE-----
MIIDCzCCAfOgAwIBAgIUJdQlkBB429XY2IdGuviOhbyVwRwwDQYJKoZIhvcNAQEL
BQAwFTETMBEGA1UEAwwKdXEtdGVzdC1jYTAeFw0yNjA3MzAxNzI2MjhaFw00NjA3
MjUxNzI2MjhaMBUxEzARBgNVBAMMCnVxLXRlc3QtY2EwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQDllTDRwVhAWNxvppkidRSeFtwvq7NDXwusYD0OJQrQ
wJhwE4exk8DCV/TeM/Tdzn+B06flKL2Kdl7NpQFVE3BMyCqMQxPftk15xBhtmqTa
EqprWfwDytvdfdN6KldEi+cW3ZTzJRL6gbgCzyTGelKqIgvKz3ZLgW3jh1HYsyg3
/v2FnOH6zhHW4kZB+vnpc3YouOBLvJSvD3d/mDQkIKjfIb310E+IZ9cLdPq6EgkE
dPmqkERxUS8jlZ5yRnYJ+jUUqcqg2/n8M7H5F9gNwstD/F5tv9Avj/tf1g12OTrh
E5CRzgM4zf87q3K7kaUt3sSQRSr9n+caeOW9Bnriuq1RAgMBAAGjUzBRMB0GA1Ud
DgQWBBSqMnOSduQQoDNIoxYX8cJYErQ4UDAfBgNVHSMEGDAWgBSqMnOSduQQoDNI
oxYX8cJYErQ4UDAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQBG
vfIS/TC907ihB30ajdbH2yuO1cQ51o1fvqOqul/aWvz5jYBBwXHmjexGESISQEjg
wKEV2t1+za13wDvkRTSiJK+Tppy7Kgnysj7zwkbcpFoXAE730dw9BKV+3S+2YsmB
894lIpwtWw6H+xulNEFbTuAuZOh6xNg4jioP7hN9r8iA38d+ak+ba9hE10hRnSJw
F23xB2CQSs+HroGRR9t+c/eexFj9BFIWlMuf2Cx/hvnmKLYBtwpx1fQPM55eSmjv
snCRaVwkeu0wVJ5ZilHtjLyoYD1LIFXycTg5Dz1n09vJimfwy+oDXz19Vx5X/3mE
YABHfQT4oBqyLB/cYMrc
-----END CERTIFICATE-----
"""


PAGE_TEXT = (
    "The tower height is 324 metres. "
    "I think the tower is the most beautiful structure ever built. "
    "The tower reportedly sways up to 9 centimetres in the wind."
)


class ParaphraseCorroborationTests(unittest.TestCase):
    """Corroboration across sources that phrase the same fact differently.

    Verbatim source-counting fired exactly never against the live web - two
    sites never phrase a definition identically - so it was mirror detection
    only. The paraphrase gate uses two independent judges (content-token
    Jaccard AND hypervector bag similarity), both thresholds measured on live
    en/simple Wikipedia pairs where the decisive boundary was J 0.250/0.207
    and H 0.311/0.231. The conjunction leans toward refusing, because a false
    corroboration inflates belief at 0.8 while a miss merely stays at 0.55.
    """

    #: The measured true-paraphrase pair, verbatim from the live boundary.
    CLAIM_A = ("The main arithmetic operations are addition, subtraction, "
               "multiplication, and division.")
    CLAIM_B = ("The four basic arithmetic operations are addition, "
               "subtraction, multiplication, and division.")
    #: Same subject, genuinely different assertion - must NOT corroborate.
    CLAIM_C = ("Arithmetic is needed in all areas of mathematics, science, "
               "and engineering.")

    def setUp(self) -> None:
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="uq_para_"))
        self.stash = ContemporaryStash(self.dir / "stash.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _status_of(self, fragment: str) -> str:
        for entry in self.stash.entries():
            if fragment in entry["claim"]:
                return entry["status"]
        raise AssertionError(f"no entry containing {fragment!r}")

    def test_a_paraphrase_from_another_site_corroborates(self) -> None:
        self.stash.add_page("http://a.example/p", "A", self.CLAIM_A)
        self.stash.add_page("http://b.example/p", "B", self.CLAIM_B)
        self.stash.analyze()
        self.assertEqual(self._status_of("main arithmetic"), "corroborated")
        self.assertEqual(self._status_of("four basic"), "corroborated")

    def test_a_different_assertion_about_the_same_subject_does_not(self) -> None:
        """The expensive error: same topic is not the same claim."""
        self.stash.add_page("http://a.example/p", "A", self.CLAIM_A)
        self.stash.add_page("http://b.example/p", "B", self.CLAIM_C)
        self.stash.analyze()
        self.assertEqual(self._status_of("main arithmetic"), "staged")
        self.assertEqual(self._status_of("needed in all areas"), "staged")

    def test_the_same_site_paraphrasing_itself_does_not(self) -> None:
        """Two sentences from one place are one witness, not two."""
        self.stash.add_page("http://a.example/p", "A",
                            self.CLAIM_A + " " + self.CLAIM_B)
        self.stash.analyze()
        self.assertEqual(self._status_of("main arithmetic"), "staged")
        self.assertEqual(self._status_of("four basic"), "staged")

    def test_verbatim_mirrors_still_corroborate(self) -> None:
        """The old path survives: identical wording from two locations."""
        self.stash.add_page("http://a.example/p", "A", self.CLAIM_A)
        self.stash.add_page("http://mirror.example/p", "M", self.CLAIM_A)
        self.stash.analyze()
        self.assertEqual(self._status_of("main arithmetic"), "corroborated")

    def test_a_corroborated_paraphrase_promotes_at_high_confidence(self) -> None:
        """The point of the whole exercise: agreement earns 0.8, not 0.55."""
        from ultraquant.memory.systematic import SystematicMemory

        self.stash.add_page("http://a.example/p", "A", self.CLAIM_A)
        self.stash.add_page("http://b.example/p", "B", self.CLAIM_B)
        self.stash.analyze()
        memory = SystematicMemory()
        entry = next(e for e in self.stash.entries()
                     if "main arithmetic" in e["claim"])
        key = self.stash.promote(entry["id"], memory)
        self.assertGreaterEqual(memory.recall_fact(key)["confidence"], 0.8)


class SubjectSupportTests(unittest.TestCase):
    """Evidence accumulated per subject, across sources, by degree.

    The live measurement behind the constants: same subject across en/simple
    Wikipedia scored 0.111-0.327; different subjects never exceeded 0.048.
    Sentence-level corroboration (even paraphrase) fired for none of the four
    subjects; subject-level fired for all four.
    """

    #: Different-aspect claims, as the live sites actually behave - one source
    #: defines, the other describes representation. No sentence pair aligns.
    EN = ("In digital imaging, a pixel is the smallest addressable element "
          "in a raster image. A pixel is commonly represented by red, green, "
          "and blue components in colour systems.")
    SIMPLE = ("A pixel is the smallest unit of a digital picture on a "
              "screen. Each pixel is a tiny square of colour on the display.")
    OTHER = ("Arithmetic is an elementary branch of mathematics. The four "
             "basic arithmetic operations are addition, subtraction, "
             "multiplication, and division.")

    def setUp(self) -> None:
        import tempfile

        self.dir = Path(tempfile.mkdtemp(prefix="uq_support_"))
        self.stash = ContemporaryStash(self.dir / "stash.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_two_sources_on_one_subject_accumulate_support(self) -> None:
        self.stash.add_page("http://a.example/p", "A", self.EN)
        self.stash.add_page("http://b.example/p", "B", self.SIMPLE)
        self.stash.analyze()
        evidence = self.stash.subject_support("pixel")
        self.assertEqual(evidence["sources"], 2)
        self.assertGreater(evidence["support"], 0.08,
                           "different aspects must still accumulate agreement")

    def test_one_source_is_no_support(self) -> None:
        self.stash.add_page("http://a.example/p", "A", self.EN)
        self.stash.analyze()
        self.assertEqual(self.stash.subject_support("pixel")["support"], 0.0)

    def test_an_unrelated_subject_gets_no_credit(self) -> None:
        """The control that makes the number mean something."""
        self.stash.add_page("http://a.example/p", "A", self.EN)
        self.stash.add_page("http://b.example/q", "B", self.OTHER)
        self.stash.analyze()
        self.assertEqual(self.stash.subject_support("pixel")["sources"], 1)
        arithmetic = self.stash.subject_support("arithmetic")
        self.assertEqual(arithmetic["sources"], 1)
        self.assertEqual(arithmetic["support"], 0.0)

    def test_research_confidence_rises_with_support(self) -> None:
        """The point: agreement by degree, not a switch."""
        import http.server
        import threading

        from ultraquant.interpreter.learning import LearningSession
        from ultraquant.interpreter.thoughts import build_session

        def serve(text: str):
            payload = (f"<html><head><title>P</title></head>"
                       f"<body><p>{text}</p></body></html>").encode("utf-8")

            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802 - http.server API
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server

        servers = [serve(self.EN), serve(self.SIMPLE)]
        try:
            session = build_session(self.dir / "home", seed=0, online=True)
            for _ in range(4):
                session.memory.remember_episode(
                    "interaction", {"text": "tell me about pixels"},
                    tags=["chat"],
                )
            learner = LearningSession(session)
            learner.survey()
            sources = [f"http://127.0.0.1:{s.server_address[1]}/{{term}}"
                       for s in servers]
            reports = learner.research(sources=sources)
            report = next(r for r in reports if r["subject"] == "pixels")
            self.assertEqual(report["outcome"], "resolved")
            self.assertIn("cross-site support", report["detail"])
            fact = (session.memory.recall_fact("pixels")
                    or session.memory.recall_fact("pixel"))
            self.assertIsNotNone(fact)
            self.assertGreater(fact["confidence"], 0.55,
                               "accumulated agreement must raise belief")
        finally:
            for server in servers:
                server.shutdown()


class ContradictionTests(unittest.TestCase):
    """Typed contradiction detection, each rule tied to a measured boundary.

    Value similarity cannot tell agreement from different-aspect for
    descriptive values (measured 0.11-0.12 vs 0.00), so the signal is the
    *type structure*: numbers compete with numbers, short identities compete
    for their slot, and descriptions never contradict.
    """

    def setUp(self) -> None:
        import tempfile

        from ultraquant.memory.systematic import SystematicMemory

        self.dir = Path(tempfile.mkdtemp(prefix="uq_contra_"))
        self.stash = ContemporaryStash(self.dir / "stash.json")
        self.memory = SystematicMemory()

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _entry(self, fragment: str) -> dict:
        for entry in self.stash.entries():
            if fragment in entry["claim"]:
                return entry
        raise AssertionError(f"no entry containing {fragment!r}")

    # -- the relation itself ------------------------------------------------

    def test_numbers_are_the_claim(self) -> None:
        from ultraquant.interpreter.stash import claim_relation

        self.assertEqual(claim_relation("The tower height is 324 metres.",
                                        "The tower height is 324 m."),
                         "agrees")
        self.assertEqual(claim_relation("The tower height is 324 metres.",
                                        "The tower height is 330 metres."),
                         "contradicts")

    def test_short_identities_compete_for_their_slot(self) -> None:
        from ultraquant.interpreter.stash import claim_relation

        self.assertEqual(claim_relation("The capital of France is Paris.",
                                        "The capital of France is Lyon."),
                         "contradicts")

    def test_descriptions_never_contradict(self) -> None:
        """The live false dispute: two agreeing definitions of arithmetic,
        phrased differently, were flagged as conflict by string inequality."""
        from ultraquant.interpreter.stash import claim_relation

        relation = claim_relation(
            "Arithmetic is the study of numbers and their operations.",
            "Arithmetic is an elementary branch of mathematics that deals "
            "with numerical operations.",
        )
        self.assertIsNone(relation)

    def test_a_measurement_and_a_description_differ_in_aspect(self) -> None:
        from ultraquant.interpreter.stash import claim_relation

        self.assertIsNone(claim_relation(
            "The tower height is 324 metres.",
            "The tower height is impressive to visitors from many countries.",
        ))

    def test_empty_content_values_yield_no_verdict_not_agreement(self) -> None:
        """Two empty hypervector bundles score 1.0 - the measured trap."""
        from ultraquant.interpreter.stash import claim_relation

        self.assertEqual(claim_relation("The span count is 4.",
                                        "The span count is 6."),
                         "contradicts")
        self.assertIsNone(claim_relation("The answer is it.",
                                         "The answer is so."))

    # -- wired into analysis -------------------------------------------------

    def test_the_live_false_dispute_is_dead(self) -> None:
        self.memory.remember_fact(
            "arithmetic", "the study of numbers and their operations", 0.7
        )
        self.stash.add_page(
            "http://a.example/x", "A",
            "Arithmetic is an elementary branch of mathematics that deals "
            "with numerical operations.",
        )
        self.stash.analyze(self.memory)
        self.assertNotEqual(self._entry("elementary branch")["status"],
                            "disputed")

    def test_web_agreement_reinforces_the_stored_fact(self) -> None:
        self.memory.remember_fact("tower height", "324 metres", 0.7)
        self.stash.add_page("http://c.example/z", "C",
                            "The tower height is 324 metres.")
        self.stash.analyze(self.memory)
        self.assertEqual(
            self.memory.recall_fact("tower height")["confidence"], 0.75
        )

    def test_a_memory_backed_claim_is_not_disputed_by_its_rival(self) -> None:
        """The rival carries the dispute; the confirmed claim does not."""
        self.memory.remember_fact("tower height", "324 metres", 0.7)
        self.stash.add_page("http://b.example/y", "B",
                            "The tower height is 330 metres.")
        self.stash.add_page("http://c.example/z", "C",
                            "The tower height is 324 metres.")
        self.stash.analyze(self.memory)
        self.assertEqual(self._entry("330")["status"], "disputed")
        self.assertNotEqual(self._entry("324")["status"], "disputed")

    def test_a_contradiction_cannot_corroborate_its_rival(self) -> None:
        """330 vs 324 shares nearly every content word and read as a
        paraphrase of its own rival until the typed relation vetoed it."""
        self.memory.remember_fact("tower height", "324 metres", 0.7)
        self.stash.add_page("http://b.example/y", "B",
                            "The tower height is 330 metres.")
        self.stash.add_page("http://c.example/z", "C",
                            "The tower height is 324 metres.")
        self.stash.analyze(self.memory)
        self.assertNotIn("paraphrased", self._entry("324")["notes"])

    def test_sources_disagreeing_with_no_memory_marks_both(self) -> None:
        """With nothing held either way, both sides become questions."""
        self.stash.add_page("http://b.example/y", "B",
                            "The pier depth is 12 metres.")
        self.stash.add_page("http://c.example/z", "C",
                            "The pier depth is 15 metres.")
        self.stash.analyze(self.memory)
        self.assertEqual(self._entry("12")["status"], "disputed")
        self.assertEqual(self._entry("15")["status"], "disputed")

    def test_disputed_sources_become_questions_for_the_human(self) -> None:
        from ultraquant.interpreter.learning import LearningSession
        from ultraquant.interpreter.thoughts import build_session

        session = build_session(self.dir / "home", seed=0)
        session.stash.add_page("http://b.example/y", "B",
                               "The pier depth is 12 metres.")
        session.stash.add_page("http://c.example/z", "C",
                               "The pier depth is 15 metres.")
        session.stash.analyze(session.memory)
        kinds = {q.kind for q in LearningSession(session).survey()}
        self.assertIn("disputed", kinds)


if __name__ == "__main__":
    unittest.main()

