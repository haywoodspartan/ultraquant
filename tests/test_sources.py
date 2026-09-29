"""Single-source distillation, using fake transport and scratch stores only."""

import io
import json
import random
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from ultraquant.distill import elicit as E, sources as S
from ultraquant.distill.targets import Target
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.systematic import SystematicMemory


def records(target, raw, teacher="t0", lineage="L0"):
    return [E.Record(teacher, lineage, "scratch.gguf", 1,
                     E.question_id(target), target.question, seed, raw, "ignored")
            for seed in E.SEEDS]


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.teacher = S.LMStudioTeacher("c4ai-command-r-08-2024", "scratch.gguf")
        self.opener = self.enterContext(mock.patch("urllib.request.urlopen"))

    def ask(self, questions=("q0", "q1"), samples=2, seeds=(7, 9)):
        return self.teacher.ask(questions, system="s0", samples=samples,
                                temperature=0.7, top_p=0.95, max_tokens=24,
                                seeds=seeds)

    def test_matrix_order_raw_contents_and_request_parameters(self):
        raw = ["  Alpha.\n", "<think>x</think>Beta", "", "é"]
        self.opener.side_effect = [io.BytesIO(json.dumps(
            {"choices": [{"message": {"content": value}}]}).encode())
            for value in raw]
        self.assertEqual(self.ask(seeds=iter((7, 9))), [raw[:2], raw[2:]])
        self.assertEqual(self.teacher.spec.name, "c4ai-command-r-08-2024")
        self.assertEqual(self.teacher.spec.gguf, Path("scratch.gguf"))
        for call, (question, seed) in zip(self.opener.call_args_list,
                                         [(q, s) for q in ("q0", "q1") for s in (7, 9)]):
            request = call.args[0]
            self.assertEqual(request.full_url,
                             "http://127.0.0.1:1234/v1/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(call.kwargs, {"timeout": 300})
            self.assertEqual(json.loads(request.data), {
                "model": self.teacher.spec.name,
                "messages": [{"role": "system", "content": "s0"},
                             {"role": "user", "content": question}],
                "temperature": 0.7, "top_p": 0.95, "max_tokens": 24, "seed": seed})
        self.assertEqual(self.opener.call_count, 4)

    def test_custom_endpoint_and_timeout(self):
        self.teacher = S.LMStudioTeacher("custom", Path("x.gguf"),
                                         "http://example.invalid/v1/", 12)
        self.opener.return_value = io.BytesIO(
            b'{"choices":[{"message":{"content":"raw"}}]}')
        self.assertEqual(self.ask(("q0",), 1, (1,)), [["raw"]])
        self.assertEqual(self.opener.call_args.args[0].full_url,
                         "http://example.invalid/v1/chat/completions")
        self.assertEqual(self.opener.call_args.kwargs, {"timeout": 12})

    def test_incomplete_seed_matrix_rejected_before_transport(self):
        with self.assertRaises(ValueError):
            self.ask(samples=3)
        self.opener.assert_not_called()
        self.assertEqual(self.ask(questions=[]), [])
        self.assertEqual(self.ask(samples=0, seeds=[]), [[], []])
        self.opener.assert_not_called()

    def test_http_and_transport_errors_propagate(self):
        for error in (urllib.error.HTTPError("fake", 500, "error", {}, None),
                      urllib.error.URLError("offline"), TimeoutError("timeout")):
            with self.subTest(error=type(error).__name__):
                self.opener.side_effect = error
                with self.assertRaises(type(error)):
                    self.ask()

    def test_invalid_json_and_response_shapes_raise(self):
        malformed = [None, [], {}, {"choices": {}}, {"choices": []},
                     {"choices": [None]}, {"choices": [{}]},
                     {"choices": [{"message": []}]},
                     {"choices": [{"message": {}}]},
                     {"choices": [{"message": {"content": None}}]},
                     {"choices": [{"message": {"content": ["raw"]}}]},
                     {"choices": [{"message": {"content": "raw"}}] * 2}]
        for payload in [b"invalid"] + [json.dumps(x).encode() for x in malformed]:
            with self.subTest(payload=payload):
                self.opener.return_value = io.BytesIO(payload)
                with self.assertRaises(ValueError):
                    self.ask()


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.scratch = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.scratch) / "nested" / "ledger.json"
        self.ledger = S.SourceLedger(self.path)

    def test_persistence_order_repeated_asks_and_source_isolation(self):
        self.assertEqual(self.ledger.asked("A"), set())
        self.assertEqual(self.ledger.history("A"), [])
        self.ledger.record("A", "q0", True)
        self.ledger.record("B", "q1", False)
        reopened = S.SourceLedger(self.path)
        reopened.record("A", "q0", False)
        self.assertEqual(self.ledger.asked("A"), {"q0"})
        self.assertEqual(reopened.history("A"), [
            {"question_id": "q0", "promoted": True},
            {"question_id": "q0", "promoted": False}])
        self.assertEqual(reopened.asked("B"), {"q1"})
        reopened.history("A")[0]["promoted"] = False
        self.assertTrue(reopened.history("A")[0]["promoted"])
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_failed_replace_preserves_previous_file(self):
        self.ledger.record("A", "q0", True)
        before = self.path.read_bytes()
        with mock.patch.object(Path, "replace", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                self.ledger.record("A", "q1", False)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.ledger.asked("A"), {"q0"})
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_frontier_exhaustion_including_empty_frontier(self):
        self.assertTrue(S.used_up(self.ledger, "A", []))
        self.assertFalse(S.used_up(self.ledger, "A", ["q0"]))
        self.ledger.record("A", "q0", False)
        self.assertTrue(S.used_up(self.ledger, "A", iter(["q0", "q0"])))
        self.assertFalse(S.used_up(self.ledger, "A", ["q0", "q1"]))
        self.assertFalse(S.used_up(self.ledger, "B", ["q0"]))

    def test_yield_uses_last_full_window_and_strict_floor(self):
        for i in range(19):
            self.ledger.record("A", "q0", i < 4)
        self.assertFalse(S.used_up(self.ledger, "A", ["unasked"]))
        self.ledger.record("A", "q0", False)
        self.assertFalse(S.used_up(self.ledger, "A", ["unasked"]))
        self.ledger.record("A", "q0", False)
        self.assertTrue(S.used_up(self.ledger, "A", ["unasked"]))
        self.assertFalse(S.used_up(self.ledger, "B", ["unasked"]))
        self.assertTrue(S.used_up(self.ledger, "A", ["unasked"], window=2, floor=0.5))
        self.assertFalse(S.used_up(self.ledger, "A", ["unasked"], window=2, floor=0))


class CalibrationItemsTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.memory = SystematicMemory(self.home / "memory.json")
        self.templates = SystematicMemory(self.home / "templates.json")
        self.stash = ContemporaryStash(self.home / "stash.json")

    def form(self, attribute, status="promoted"):
        subject = "seed"
        title = f"[{attribute}] <{subject}> {{tag}}"
        entry = self.stash.add_claim(
            f"https://distill.invalid/scratch/{attribute}:{subject}", title, "stored",
            provenance={"run_id": "scratch", "question_id": f"{attribute}:{subject}"},
            fields={"key": attribute, "value": "stored", "subject": subject,
                    "attribute": attribute})
        if status == "promoted":
            self.stash.promote(entry, self.templates, force=True)
        elif status == "rejected":
            self.stash.reject(entry)

    def test_only_held_structured_facts_with_stored_forms_are_sampled(self):
        self.form("B")
        self.form("A")
        self.form("staged", "staged")
        self.form("rejected", "rejected")
        expected = []
        for attribute in ("B", "A", "missing", "staged", "rejected"):
            for subject in ("Zulu", "Alpha", "Bravo"):
                value = f"value-{attribute}-{subject}"
                self.memory.remember_fact(f"{subject}-{attribute}", value,
                                          subject=subject, attribute=attribute)
                if attribute in ("A", "B"):
                    expected.append((Target(attribute, subject,
                                            f"[{attribute}] <{subject}> {{tag}}", attribute),
                                     value))
        self.memory.remember_fact("unstructured", "value")
        self.memory.remember_fact("partial", "value", subject="partial")
        self.memory.remember_fact("partial2", "value", attribute="A")
        expected.sort(key=lambda pair: (pair[0].attribute, pair[0].subject))
        state = random.getstate()
        self.assertEqual(S.calibration_items(self.memory, self.stash, k=4),
                         random.Random(155).sample(expected, 4))
        self.assertEqual(random.getstate(), state)
        self.memory.save()
        reopened = SystematicMemory(self.home / "memory.json")
        self.assertEqual(S.calibration_items(reopened, ContemporaryStash(
            self.home / "stash.json"), k=4), random.Random(155).sample(expected, 4))
        self.assertEqual(S.calibration_items(self.memory, self.stash),
                         random.Random(155).sample(expected, len(expected)))
        self.assertEqual(S.calibration_items(self.memory, self.stash, k=4, seed=7),
                         random.Random(7).sample(expected, 4))
        self.assertEqual(S.calibration_items(self.memory, self.stash, k=0), [])

    def test_empty_store(self):
        self.assertEqual(S.calibration_items(self.memory, self.stash), [])


class DecisionTests(unittest.TestCase):
    def test_default_and_single_lineage_thresholds(self):
        target = Target("slot", "subject", "q0", "attribute")
        key = E.question_id(target)
        one = records(target, "Alpha")
        self.assertEqual(E.decide(one, [target]), {key: None})
        self.assertEqual(E.decide(one, [target]), E.decide(one, [target], min_lineages=2))
        self.assertEqual(E.decide(one, [target], min_lineages=1), {key: "alpha"})
        self.assertEqual(S.single_source_decide(one, [target]), {key: "alpha"})
        self.assertEqual(S.single_source_decide(one[:2], [target]), {key: None})
        self.assertEqual(E.decide(one + records(target, "Alpha", "t1", "L1"),
                                  [target]), {key: "alpha"})
        for lineage in ("L0", "L1"):
            self.assertEqual(S.single_source_decide(
                one + records(target, "Beta", "t1", lineage), [target]), {key: None})

    def test_calibration_counts_normalized_agreement_and_wilson(self):
        items = [Target("slot", f"s{i}", f"q{i}", "attribute") for i in range(4)]
        pairs = list(zip(items, ["The Álpha", "Beta", "Gamma", "Delta"]))
        samples = (records(items[0], "Alpha") + records(items[1], "Named Beta")
                   + records(items[2], "Wrong") + records(items[3], "UNKNOWN"))
        report = S.calibrate(iter(samples), iter(pairs))
        self.assertEqual({k: report[k] for k in ("asked", "promoted", "right")},
                         {"asked": 4, "promoted": 3, "right": 2})
        self.assertAlmostEqual(report["wilson_lower"], 0.2076596, places=6)
        self.assertEqual(S.calibrate([], pairs),
                         {"asked": 4, "promoted": 0, "right": 0, "wilson_lower": 0.0})
        self.assertEqual(S.calibrate([], []),
                         {"asked": 0, "promoted": 0, "right": 0, "wilson_lower": 0.0})
        self.assertAlmostEqual(S.calibrate(samples[:5], pairs[:1])["wilson_lower"],
                               0.2065493, places=6)
        self.assertEqual(S.calibrate(records(items[0], "Wrong"), pairs[:1])[
            "wilson_lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
