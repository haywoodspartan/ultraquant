"""Session orchestration with fake transport, teachers, and scratch libraries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from ultraquant.distill import frontier, session, sources, targets
from ultraquant.distill.teachers import TeacherSpec
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def model(name, context=4096, parallel=None, ttl=None, kind="llm"):
    return dict(identifier=name, type=kind, contextLength=context,
                parallel=parallel, ttlMs=ttl)


class FakeRun:
    def __init__(self, loaded=()):
        self.loaded = deepcopy(list(loaded))
        self.calls = []

    def __call__(self, args, *, capture_output, text):
        assert capture_output is True and text is True
        self.calls.append(args)
        if args[1] == "ps":
            return subprocess.CompletedProcess(args, 0, json.dumps(self.loaded), "")
        if args[1] == "unload":
            self.loaded = [m for m in self.loaded if m["identifier"] != args[2]]
        elif args[1] == "load":
            def flag(name, default=None):
                return args[args.index(name) + 1] if name in args else default
            ttl = flag("--ttl")
            parallel = flag("--parallel")
            self.loaded.append(model(args[2], int(flag("--context-length")),
                                     int(parallel) if parallel is not None else None,
                                     float(ttl) * 1000 if ttl is not None else None))
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, "", "")


class SwapperTests(unittest.TestCase):
    def test_snapshot_and_load_preserve_embeddings_and_already_loaded_source(self):
        embedding = model("embed", 2048, ttl=3600000, kind="embedding")
        run = FakeRun([embedding, model("old"), model("wanted")])
        swapper = session.LMStudioSwapper(Path("fake.exe"), run=run)
        self.assertEqual(swapper.snapshot(), run.loaded)
        swapper.load("wanted", 8192)
        self.assertEqual(run.calls, [["fake.exe", "ps", "--json"],
                                     ["fake.exe", "ps", "--json"],
                                     ["fake.exe", "unload", "old"]])
        self.assertEqual(run.loaded, [embedding, model("wanted")])
        swapper.load("next", 8192)
        self.assertEqual(run.calls[-2:], [["fake.exe", "unload", "wanted"],
                         ["fake.exe", "load", "next", "--context-length", "8192", "-y"]])
        self.assertEqual(run.loaded[0], embedding)

    def test_restore_unchanged_ignores_transient_fields(self):
        wanted = [model("source", parallel=4), model("embed", kind="embedding")]
        run = FakeRun(wanted)
        run.loaded[0].update(status="idle", lastUsedTime=42)
        session.LMStudioSwapper("fake", run).restore(wanted)
        self.assertEqual(run.calls, [["fake", "ps", "--json"]])

    def test_restore_unloads_extras_loads_missing_and_reloads_changed_settings(self):
        for field, value in (("contextLength", 123), ("parallel", 2), ("ttlMs", 2000)):
            with self.subTest(field=field):
                wanted = [model("original", 8192, 4, 1500),
                          model("missing", 2048, ttl=0, kind="embedding")]
                changed = dict(wanted[0], **{field: value})
                run = FakeRun([changed, model("extra", kind="embedding")])
                session.LMStudioSwapper("fake", run).restore(wanted)
                self.assertEqual(run.calls, [
                    ["fake", "ps", "--json"],
                    ["fake", "unload", "original"], ["fake", "unload", "extra"],
                    ["fake", "load", "original", "--context-length", "8192",
                     "--parallel", "4", "--ttl", "1.5", "-y"],
                    ["fake", "load", "missing", "--context-length", "2048", "--ttl", "0", "-y"]])

    def test_cli_failure_is_not_silently_accepted(self):
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 1, "", "failed"))
        with self.assertRaises(subprocess.CalledProcessError):
            session.LMStudioSwapper("fake", run).snapshot()

    def test_restore_retains_millisecond_precision_in_long_ttl(self):
        run = FakeRun()
        session.LMStudioSwapper("fake", run).restore([model("original", ttl=3600001)])
        self.assertEqual(run.calls[-1], ["fake", "load", "original", "--context-length",
                                        "4096", "--ttl", "3600.001", "-y"])


class FakeTeacher:
    def __init__(self, source, events):
        self.spec = TeacherSpec(source.name, Path(source.gguf))
        self.events = events

    def ask(self, questions, *, samples, **kwargs):
        self.events.append(("calibrate", self.spec.name, list(questions)))
        return [["Alpha"] * samples for _ in questions]


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.home / "library"
        self.root.mkdir()
        self.backups = self.home / "backups"
        self.report_path = self.home / "report.json"
        self.events = []
        gguf = self.home / "fake.gguf"
        name, value = b"general.name", b"Fake teacher"
        gguf.write_bytes(struct.pack("<4sIQQQ", b"GGUF", 3, 0, 1, len(name))
                         + name + struct.pack("<IQ", 8, len(value)) + value)
        self.plan = [session.SourcePlan(n, str(gguf), 4096)
                     for n in ("first", "second/model", "third")]
        self.memory = SystematicMemory(self.root / "memory.json")
        self.memory.remember_fact("capital of somewhere", "Alpha", 0.99,
                                  subject="Somewhere", attribute="capital")
        self.memory.save()
        self.stash = ContemporaryStash(self.root / "stash.json")
        self.stash.add_claim("https://distill.invalid/test/capital:Somewhere",
                             "What is the capital of Somewhere?",
                             "The capital of Somewhere is Alpha.",
                             provenance={"teachers": [s.name for s in self.plan],
                                         "run_id": "test", "question_id": "capital:Somewhere"},
                             fields={"key": "capital of somewhere", "subject": "Somewhere",
                                     "attribute": "capital", "value": "Alpha"})
        self.stash._entries[1]["status"] = "promoted"
        self.stash.save()
        self.pairs = [(targets.Target("capital", "Somewhere",
                                     "What is the capital of Somewhere?", "capital"), "Alpha")]
        self.loaded = [model("original", 8192, 4, 3600000)]
        self.swapper = mock.Mock()
        self.swapper.snapshot.side_effect = lambda: self.event("snapshot", deepcopy(self.loaded))
        self.swapper.load.side_effect = lambda name, context: self.event(("load", name))
        self.swapper.restore.side_effect = lambda snapshot: self.event(("restore", snapshot))
        # Unexpected external calls fail instead of reaching a process or network.
        self.enterContext(mock.patch("subprocess.run", side_effect=AssertionError("external process")))
        self.enterContext(mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")))

    def event(self, event, result=None):
        self.events.append(event)
        return result

    def run_session(self, **kwargs):
        return session.run_session(self.root, self.plan, swapper=self.swapper,
                                   teacher_factory=lambda s: FakeTeacher(s, self.events),
                                   backup_dir=self.backups, report_path=self.report_path,
                                   **kwargs)

    def test_load_plan_default_custom_and_frozen(self):
        plan = session.load_plan()
        self.assertEqual([s.name for s in plan], ["c4ai-command-r-08-2024",
                         "qwen/qwen3.8-27b", "cydonia-v1.3-magnum-v4-22b"])
        self.assertTrue(all(s.context_length == 4096 and s.gguf.endswith(".gguf") for s in plan))
        path = self.home / "plan.json"
        path.write_text(json.dumps({"context_length": 128, "sequence": [
            {"name": "custom", "gguf": "model.gguf"}]}), encoding="utf-8")
        self.assertEqual(session.load_plan(path), [session.SourcePlan("custom", "model.gguf", 128)])
        with self.assertRaises(FrozenInstanceError):
            plan[0].name = "changed"

    def test_backup_is_complete_unique_and_outside_root(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "data").write_bytes(b"\x00\xff")
        (self.root / "empty").mkdir()
        first = session.backup(self.root, self.backups)
        second = session.backup(self.root, self.backups)
        self.assertNotEqual(first, second)
        self.assertEqual(tree(first), tree(self.root))
        self.assertTrue((first / "empty").is_dir())
        self.assertTrue(first.name.startswith("library-"))
        with self.assertRaises(ValueError):
            session.backup(self.root, self.root / "backups")

    def test_order_backup_first_totals_and_pairs_drawn_once(self):
        before = tree(self.root)
        real_backup = session.backup

        def backup(root, directory):
            self.assertEqual(self.events, [])
            self.assertEqual(tree(root), before)
            self.event("backup")
            return real_backup(root, directory)

        counts = {}

        def round(memory, stash, teacher, ledger, source, **kwargs):
            self.event(("round", source))
            counts[source] = counts.get(source, 0) + 1
            self.assertGreater(kwargs["confidence"], 0)
            self.assertTrue(kwargs["run_id"].endswith(f"-{source}-{counts[source]}"))
            self.assertIn(self.backups, Path(kwargs["records_path"]).parents)
            self.assertEqual(kwargs["approver"].memory, memory)
            memory.remember_fact("new", source, 0.9)
            return dict(asked=5, filed=2, agreed=1, contested=1, revised=1,
                        used_up=counts[source] == 2)

        with mock.patch.object(session, "backup", side_effect=backup), \
                mock.patch.object(sources, "calibration_items", return_value=self.pairs) as pairs, \
                mock.patch.object(frontier, "study_round", side_effect=round):
            report = self.run_session()
        pairs.assert_called_once()
        self.assertEqual(pairs.call_args.kwargs, dict(k=40, seed=155))
        expected = ["backup", "snapshot"]
        for source in self.plan:
            expected += [("load", source.name), ("calibrate", source.name, [self.pairs[0][0].question]),
                         ("round", source.name), ("round", source.name)]
        expected += [("restore", self.loaded)]
        self.assertEqual(self.events, expected)
        self.assertEqual(tree(Path(report["backup"])), before)
        self.assertEqual(report["status"], "complete")
        self.assertIsNone(report["failed"])
        self.assertIsNone(report["error"])
        self.assertEqual(json.loads(self.report_path.read_text()), report)
        for entry in report["sources"]:
            self.assertEqual(entry["co_distilled"], 1)
            self.assertEqual(entry["calibration"]["right"], 1)
            self.assertEqual(entry["calibration"]["decided"], 1)
            self.assertEqual(entry["totals"], dict(asked=10, filed=4, agreed=2, contested=2, revised=2))

    def test_raising_round_restores_saves_reports_and_reraises_same_error(self):
        fault = RuntimeError("round failed")

        def round(memory, stash, teacher, ledger, source, **kwargs):
            if source == self.plan[1].name:
                memory.remember_fact("partial", "retained", 0.9)
                raise fault
            return dict(asked=1, filed=1, agreed=0, contested=0, revised=0, used_up=True)

        with mock.patch.object(frontier, "study_round", side_effect=round), \
                mock.patch.object(sources, "calibration_items", side_effect=AssertionError("redrawn")):
            with self.assertRaises(RuntimeError) as raised:
                self.run_session(pairs=self.pairs)
        self.assertIs(raised.exception, fault)
        self.swapper.restore.assert_called_once_with(self.loaded)
        report = json.loads(self.report_path.read_text())
        self.assertEqual((report["status"], report["failed"], report["error"]),
                         ("failed", self.plan[1].name, repr(fault)))
        self.assertTrue(Path(report["backup"]).is_dir())
        self.assertEqual(SystematicMemory(self.memory.path).recall_fact("partial")["value"], "retained")

    def test_max_rounds_and_restore_failure_still_write_report(self):
        with mock.patch.object(frontier, "study_round", return_value={"used_up": False}) as round:
            report = self.run_session(pairs=self.pairs, max_rounds=2)
        self.assertEqual(round.call_count, 6)
        self.swapper.restore.side_effect = RuntimeError("restore failed")
        with self.assertRaisesRegex(RuntimeError, "restore failed"):
            self.run_session(pairs=self.pairs, max_rounds=0)
        self.assertIn("restore failed", json.loads(self.report_path.read_text())["error"])

    def test_co_distilled_resolves_ids_names_and_subject_attribute(self):
        self.stash._entries[1]["provenance"] = {"teacher_ids": ["weights"]}
        with mock.patch.object(sources, "identity", return_value="weights"):
            self.assertEqual(session._co_distilled(self.stash, self.pairs, "alias"), 1)
            other = [(targets.Target("capital", "Elsewhere", "q", "capital"), "Alpha")]
            self.assertEqual(session._co_distilled(self.stash, other, "alias"), 0)
        self.stash._entries[1]["status"] = "staged"
        self.assertEqual(session._co_distilled(self.stash, self.pairs, "weights"), 0)

    def test_dry_run_writes_nothing_with_or_without_vault(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                if sharded:
                    shards = FactShards(ShardVault(self.root / "vault"))
                    memory = SystematicMemory(self.memory.path, shards=shards)
                    memory.save()
                before = tree(self.home)
                output = io.StringIO()
                with mock.patch.object(session, "load_plan", return_value=self.plan) as plan, \
                        mock.patch.object(session, "LMStudioSwapper", return_value=self.swapper), \
                        mock.patch.object(session, "backup", side_effect=AssertionError("backup")), \
                        mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), \
                        mock.patch.object(Path, "write_text", side_effect=AssertionError("write")), \
                        mock.patch.object(SystematicMemory, "save", side_effect=AssertionError("save")), \
                        mock.patch("sys.stdout", output):
                    self.assertEqual(session.main(["--root", str(self.root), "--dry-run",
                                                   "--backup-dir", str(self.backups),
                                                   "--report", str(self.report_path)]), 0)
                plan.assert_called_once_with()
                data = json.loads(output.getvalue())
                self.assertEqual(data["snapshot"], self.loaded)
                self.assertEqual(data["pairs"], 1)
                self.assertEqual([s["co_distilled"] for s in data["sources"]], [1, 1, 1])
                self.assertEqual(tree(self.home), before)
                self.swapper.load.assert_not_called()
                self.swapper.restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
