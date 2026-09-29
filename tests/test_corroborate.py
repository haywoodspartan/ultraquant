"""Second-source checks with fake teachers and temporary stores only."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_frontier import FakeTeacher
from ultraquant.distill import corroborate, elicit, frontier, sources, targets
from ultraquant.distill.teachers import TeacherSpec
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class ValuesAgreeTests(unittest.TestCase):
    def test_numbers_at_coarser_precision(self):
        for a, b, expected in (
                ("12", "12.011", True), ("196.97", "196.966569", True),
                ("95.96", "95.95", False), ("12.0", "12.05", False),
                ("12.1", "12.05", True), ("-12.1", "-12.05", True),
                ("-12", "12", False), ("1,234.6 u", "1234.56", True),
                ("65", "69", False), ("32.065", "32.06", False),
                ("2 then 99", "2 then 44", True),
                ('<think>999</think>**Answer:** "12.011"', "12", True),
                ("123456789012345678901234567890", "123456789012345678901234567890.1", True)):
            with self.subTest(a=a, b=b):
                self.assertIs(corroborate.values_agree(a, b), expected)
                self.assertIs(corroborate.values_agree(b, a), expected)

    def test_strings_and_mixed_forms(self):
        for a, b, expected in (("The Paris.", "Paris", True),
                               ("Au", "au", True), ("No", "No", True),
                               ("Paris", "Lyon", False), ("12", "twelve", False),
                               ("12 red", "red", True)):
            with self.subTest(a=a, b=b):
                self.assertIs(corroborate.values_agree(a, b), expected)


class TeacherOptionsTests(unittest.TestCase):
    def test_data_default_absent_model_and_explicit_options(self):
        data = json.loads((Path(sources.__file__).with_name("data") /
                           "sources.json").read_text(encoding="utf-8"))
        model, defaults = next(iter(data.items()))
        explicit = {"temperature": 0.1, "max_tokens": 40}
        for name, options, expected in ((model, None, defaults),
                                         ("unlisted", None, {}),
                                         (model, explicit, dict(explicit)),
                                         (model, {}, {})):
            with self.subTest(model=name, options=options):
                teacher = sources.LMStudioTeacher(name, "fake.gguf", options=options)
                self.assertEqual(teacher.options, expected)
                if options is not None:
                    self.assertIsNot(teacher.options, options)
                with mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k:
                                io.BytesIO(b'{"choices":[{"message":{"content":"raw"}}]}')) as opener:
                    self.assertEqual(teacher.ask(
                        ["q", "r"], system="s", samples=2, temperature=0.7,
                        top_p=0.95, max_tokens=24, seeds=(1, 2)), [["raw"] * 2] * 2)
                for call, (question, seed) in zip(opener.call_args_list,
                                                  [(q, s) for q in ("q", "r") for s in (1, 2)]):
                    self.assertEqual(json.loads(call.args[0].data), {
                        "model": name, "messages": [{"role": "system", "content": "s"},
                                                     {"role": "user", "content": question}],
                        "temperature": 0.7, "top_p": 0.95, "max_tokens": 24,
                        "seed": seed, **expected})


class CorroborateTests(unittest.TestCase):
    sharded = False

    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shards = FactShards(ShardVault(self.home / "vault")) if self.sharded else None
        self.memory = SystematicMemory(self.home / "memory.json", shards=shards)
        self.stash = ContemporaryStash(self.home / "stash.json")
        self.ledger = sources.SourceLedger(self.home / "ledger.json")
        self.teacher = FakeTeacher(self.home)
        self.teacher.spec = TeacherSpec("second", self.teacher.spec.gguf)
        self.enterContext(mock.patch("urllib.request.urlopen",
                                     side_effect=AssertionError("Network forbidden")))
        self.enterContext(mock.patch.object(sources.LMStudioTeacher, "ask",
                                           side_effect=AssertionError("Live teacher forbidden")))

    def claim(self, subject="Carbon", value="12", attribute="atomic mass",
              teachers=("first",), learned=True):
        fields = dict(key=f"{attribute} of {subject.lower()}", value=value,
                      subject=subject, attribute=attribute)
        provenance = {"run_id": "first", "question_id": f"mass:{subject}"}
        if teachers is not None:
            provenance.update(teachers=list(teachers), lineages=list(teachers))
        title = f"Tell me the {attribute} of {subject}." if learned else "Untitled"
        entry_id = self.stash.add_claim(
            f"https://distill.invalid/first/{subject}", title,
            f"The {attribute} of {subject} is {value}.",
            provenance=provenance, fields=fields)
        self.stash.promote(entry_id, self.memory)
        return entry_id

    def run_checks(self, source="second"):
        return corroborate.corroborate(
            self.memory, self.stash, self.teacher, self.ledger, source,
            records_path=self.home / "checks.jsonl", run_id="run")

    def answer(self, entry_id, raw):
        check = self.stash.get(entry_id)["fields"]
        target = corroborate._forward_target(self.stash, check)
        self.teacher.replies[target.question] = raw if isinstance(raw, list) else [raw] * 5
        return target

    def test_claim_filters_latest_entry_and_held_value(self):
        old = self.claim()
        latest = self.claim(value="12.011")
        self.memory.remember_fact("atomic mass of carbon", "12", subject="Carbon",
                                  attribute="atomic mass")
        self.claim("Own", teachers=("second",))
        self.claim("Multiple", teachers=("first", "third"))
        self.claim("User", teachers=None)
        self.claim("Empty", teachers=())
        stale = self.claim("Stale", "99")
        self.stash._entries[stale]["fields"]["value"] = "55"
        staged = self.claim("Staged")
        self.stash._entries[staged]["status"] = "staged"
        for field in ("key", "value", "subject", "attribute"):
            incomplete = self.claim(f"Missing {field}")
            del self.stash._entries[incomplete]["fields"][field]
        absent = self.claim("Absent")
        self.stash._entries[absent]["fields"]["key"] = "not held"
        checks = corroborate.claims_to_check(self.memory, self.stash, "second")
        self.assertEqual(checks, [{"entry_id": latest, "key": "atomic mass of carbon",
                                  "value": "12", "subject": "Carbon", "attribute": "atomic mass"}])
        self.assertNotEqual(old, latest)

    def test_batch_verdicts_persistence_queue_and_once_per_source(self):
        agreed = self.claim()
        contested = self.claim("Osmium", "225.87")
        undecided = self.claim("Mystery", "44")
        self.answer(agreed, ["12.011", "**12.011**", "12.011 u", "12.011", "UNKNOWN"])
        self.answer(contested, "190.23")
        self.answer(undecided, "UNKNOWN")
        before = {key: self.memory.recall_fact(key) for key in self.memory.fact_keys()}
        with mock.patch.object(corroborate, "values_agree", wraps=corroborate.values_agree) as agree, \
                mock.patch.object(corroborate, "contest", wraps=corroborate.contest) as contest, \
                mock.patch.object(sources, "single_source_decide", wraps=sources.single_source_decide) as decide:
            self.assertEqual(self.run_checks(), dict(checked=3, agreed=1, contested=1,
                                                     undecided=1, asked=3))
        self.assertGreaterEqual(agree.call_count, 2)
        contest.assert_called_once()
        decide.assert_called_once()
        self.assertEqual(contest.call_args.args[1]["answer"], "190.23")
        self.assertEqual(len(self.teacher.calls), 1)
        self.assertEqual(len(self.teacher.calls[0][0]), 3)
        reloaded = ContemporaryStash(self.stash.path)
        provenance = reloaded.get(agreed)["provenance"]
        self.assertEqual(provenance["teachers"], ["first", "second"])
        self.assertEqual(provenance["lineages"], ["first", "second"])
        self.assertEqual(provenance["run_id"], "first")
        reloaded.add_teacher(agreed, "second")
        self.assertEqual(reloaded.get(agreed)["provenance"], provenance)
        self.assertEqual(before, {key: self.memory.recall_fact(key) for key in self.memory.fact_keys()})
        rows = sources.SourceLedger(self.ledger.path).history("second")
        self.assertEqual([r["check"]["verdict"] for r in rows], ["agreed", "contested", "undecided"])
        self.assertEqual([r["promoted"] for r in rows], [True, False, False])
        self.assertEqual(rows[0]["check"]["answer"], "12.011")
        self.assertIsNone(rows[2]["check"]["answer"])
        self.assertEqual(rows[1]["queued"], dict(key="atomic mass of osmium", value="225.87",
                                               subject="Osmium", attribute="atomic mass", contest="190.23"))
        self.assertIn(("atomic mass of osmium", "225.87"),
                      frontier.verification_queue(self.memory, self.ledger))
        self.assertEqual(self.run_checks(), dict(checked=0, agreed=0, contested=0, undecided=0, asked=0))
        self.assertEqual(len(self.teacher.calls), 1)
        self.assertEqual(self.run_checks("third")["checked"], 2)

    def test_learned_and_seed_forward_forms_and_existing_ledger(self):
        learned = self.claim()
        seed = self.claim("Iron", "26", attribute="Atomic Number", learned=False)
        learned_target = self.answer(learned, "12.011")
        seed_target = self.answer(seed, "26")
        self.assertEqual(learned_target, targets.Target(
            "mass", "Carbon", "Tell me the atomic mass of Carbon.", "atomic mass"))
        self.assertEqual(seed_target, targets.Target(
            "atomic number", "Iron", "What is the Atomic Number of Iron?", "Atomic Number"))
        self.ledger.path.write_text(json.dumps({"second": [{
            "question_id": elicit.question_id(learned_target), "promoted": False}]}))
        self.assertEqual(self.run_checks()["checked"], 1)
        self.assertEqual(self.teacher.calls[0][0], [seed_target.question])
        self.assertNotIn("check", self.ledger.history("second")[0])

    def test_contest_seam_can_suppress_queue(self):
        self.answer(self.claim(), "19")
        with mock.patch.object(corroborate, "contest", return_value=None) as contest:
            self.assertEqual(self.run_checks()["contested"], 1)
        contest.assert_called_once()
        self.assertNotIn("queued", self.ledger.history("second")[0])
        self.assertEqual(self.memory.recall_fact("atomic mass of carbon")["value"], "12")

    def test_study_checks_before_pending_then_is_used_up(self):
        target = self.answer(self.claim(), "12.011")
        pending = frontier.pending

        def after_checks(*args):
            self.assertEqual(self.teacher.calls[0][0], [target.question])
            self.assertEqual(self.ledger.history("second")[0]["check"]["verdict"], "agreed")
            return pending(*args)

        with mock.patch.object(frontier, "pending", side_effect=after_checks), \
                mock.patch.object(corroborate, "corroborate", wraps=corroborate.corroborate) as check:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, self.ledger, "second",
                confidence=0.99, run_id="study", records_path=self.home / "study.jsonl")
        check.assert_called_once()
        self.assertEqual({key: result[key] for key in ("checked", "agreed", "contested", "asked", "used_up")},
                         dict(checked=1, agreed=1, contested=0, asked=1, used_up=True))
        result = frontier.study_round(
            self.memory, self.stash, self.teacher, self.ledger, "second",
            confidence=0.99, run_id="again", records_path=self.home / "again.jsonl")
        self.assertEqual(result["asked"], 0)
        self.assertTrue(result["used_up"])
        self.assertEqual(len(self.teacher.calls), 1)

    def test_first_source_asks_only_its_existing_frontier(self):
        self.claim(teachers=("second",))
        target = targets.Target("mass", "Iron", "Existing frontier question", "atomic mass")
        self.teacher.replies[target.question] = ["UNKNOWN"] * 5
        with mock.patch.object(frontier, "pending", return_value=[target]), \
                mock.patch.object(frontier, "propose") as propose:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, self.ledger, "second",
                confidence=0.99, run_id="first", records_path=self.home / "first.jsonl")
        propose.assert_not_called()
        self.assertEqual(self.teacher.calls, [([target.question], dict(
            system=elicit.SYSTEM, samples=elicit.SAMPLES, temperature=elicit.TEMPERATURE,
            top_p=elicit.TOP_P, max_tokens=elicit.MAX_TOKENS, seeds=elicit.SEEDS))])
        self.assertEqual(result["asked"], 1)
        self.assertEqual(result["checked"], 0)
        self.assertEqual(self.ledger.history("second"), [{
            "question_id": elicit.question_id(target), "promoted": False}])


class ShardedCorroborateTests(CorroborateTests):
    sharded = True


if __name__ == "__main__":
    unittest.main()
