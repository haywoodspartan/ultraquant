"""Session orchestration with fake transport, teachers, and scratch libraries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
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

    def __call__(self, args, *, capture_output, text, encoding, errors):
        assert capture_output is True and text is True and encoding == "utf-8"
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
            loaded = model(args[2], int(flag("--context-length")),
                           int(parallel) if parallel is not None else None,
                           float(ttl) * 1000 if ttl is not None else None)
            # `--gpu off` places a model CPU-only (§11.168); a plain load has no flag.
            if "--gpu" in args:
                loaded["gpu"] = flag("--gpu")
            self.loaded.append(loaded)
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, "", "")


PS = ("ps", "--json")


def cpu_load(name, context=4096):
    return ("load", name, "--gpu", "off", "--context-length", str(context), "-y")


class SwapperTests(unittest.TestCase):
    def test_lms_output_is_read_as_utf8(self):
        # §11.166 live: `lms load` prints UTF-8 progress (U+258F is E2 96 8F)
        # and the Windows code page has no character for 0x8F, so a reader
        # thread raised and the output was lost. `ps --json` shares the path.
        script = ("import sys; sys.stdout.buffer.write("
                  "'[{\"identifier\": \"m\u258f\", \"type\": \"llm\"}]'.encode('utf-8'))")

        def run(args, **kwargs):
            return subprocess.run([sys.executable, "-c", script], **kwargs)
        swapper = session.LMStudioSwapper("lms", run=run)
        self.assertEqual(swapper.snapshot(), [{"identifier": "m▏", "type": "llm"}])

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

    def test_load_alongside_shares_a_loaded_model_issuing_only_ps(self):
        user = [model("mine", 128000, 4), model("embed", 2048, kind="embedding")]
        run = FakeRun(user)
        self.assertEqual(session.LMStudioSwapper("fake", run).load_alongside("mine", 4096),
                         "shared")
        self.assertEqual(run.calls, [["fake", "ps", "--json"]])
        self.assertEqual(run.loaded, user)

    def test_load_alongside_loads_cpu_only_beside_everything_and_never_unloads(self):
        user = [model("mine", 128000, 4), model("embed", 2048, ttl=3600000, kind="embedding")]
        run = FakeRun(user)
        swapper = session.LMStudioSwapper("fake", run)
        self.assertEqual(swapper.load_alongside("source", 8192), "cpu")
        self.assertEqual(run.calls, [["fake", "ps", "--json"],
                                     ["fake", "load", "source", "--gpu", "off",
                                      "--context-length", "8192", "-y"]])
        # A second source joins the first: unlike load, nothing is unloaded.
        self.assertEqual(swapper.load_alongside("next", 4096), "cpu")
        self.assertNotIn("unload", [call[1] for call in run.calls])
        self.assertEqual(run.loaded, user + [dict(model("source", 8192), gpu="off"),
                                             dict(model("next"), gpu="off")])

    def test_unload_unloads_only_the_named_model(self):
        run = FakeRun([model("mine", 128000, 4), model("source")])
        session.LMStudioSwapper("fake", run).unload("source")
        self.assertEqual(run.calls, [["fake", "unload", "source"]])
        self.assertEqual(run.loaded, [model("mine", 128000, 4)])


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

    def lms(self, loaded, fail=()):
        """Swap through a real swapper over FakeRun, logging its commands as events.

        A command in ``fail`` fails once, as lms would, and changes nothing.
        """
        run, fail = FakeRun(loaded), set(fail)

        def logged(args, **kwargs):
            command = tuple(args[1:])
            self.event(command)
            if command in fail:
                fail.remove(command)
                return subprocess.CompletedProcess(args, 1, "", "failed")
            return run(args, **kwargs)
        self.swapper = session.LMStudioSwapper("lms", logged)
        return run

    def study_round(self, memory, stash, teacher, ledger, source, **kwargs):
        self.event(("round", source))
        return {"used_up": True}

    def studied(self, name):
        return [("calibrate", name, [self.pairs[0][0].question]), ("round", name)]

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

    def test_load_device_default_absent_and_bad(self):
        self.assertEqual(session.load_device(), "cpu")
        self.assertEqual(set(json.loads(session.PLAN_PATH.read_text(encoding="utf-8"))),
                         {"cli", "context_length", "device", "sequence"})
        path = self.home / "plan.json"
        path.write_text(json.dumps({"context_length": 128, "sequence": []}), encoding="utf-8")
        self.assertEqual(session.load_device(path), "gpu")
        for device in ("cpu", "gpu"):
            path.write_text(json.dumps({"device": device}), encoding="utf-8")
            self.assertEqual(session.load_device(path), device)
        for device in ("GPU", "cuda", "", None, 0, ["cpu"]):
            with self.subTest(device=device):
                path.write_text(json.dumps({"device": device}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    session.load_device(path)

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
                self.assertEqual(data["device"], "cpu")
                self.assertEqual(data["snapshot"], self.loaded)
                self.assertEqual(data["pairs"], 1)
                self.assertEqual([s["co_distilled"] for s in data["sources"]], [1, 1, 1])
                self.assertEqual(tree(self.home), before)
                self.swapper.load.assert_not_called()
                self.swapper.restore.assert_not_called()

    def test_cpu_shares_a_loaded_source_and_unloads_each_it_loaded_before_the_next(self):
        user = [model("first", 128000, 4), model("embed", 2048, ttl=3600000, kind="embedding")]
        run = self.lms(user)
        with mock.patch.object(frontier, "study_round", side_effect=self.study_round):
            report = self.run_session(pairs=self.pairs, device="cpu")
        expected = [PS, PS] + self.studied("first")
        for name in ("second/model", "third"):
            expected += [PS, cpu_load(name)] + self.studied(name) + [("unload", name)]
        # The restore finds the models as they were and issues nothing but ps.
        self.assertEqual(self.events, expected + [PS])
        self.assertEqual(run.loaded, user)
        self.assertEqual(report["status"], "complete")
        self.assertEqual([(e["name"], e["device"]) for e in report["sources"]],
                         [("first", "shared"), ("second/model", "cpu"), ("third", "cpu")])

    def test_cpu_source_that_raises_is_unloaded_and_its_error_reraised(self):
        real_ask = FakeTeacher.ask
        for stage in ("calibration", "round"):
            with self.subTest(stage=stage):
                self.events.clear()
                fault = RuntimeError(f"{stage} failed")
                user = [model("first", 128000, 4)]
                run = self.lms(user)

                def ask(teacher, questions, **kwargs):
                    if stage == "calibration" and teacher.spec.name == "second/model":
                        raise fault
                    return real_ask(teacher, questions, **kwargs)

                def study_round(memory, stash, teacher, ledger, source, **kwargs):
                    self.event(("round", source))
                    if source == "second/model":
                        raise fault
                    return {"used_up": True}

                with mock.patch.object(FakeTeacher, "ask", ask), \
                        mock.patch.object(frontier, "study_round", side_effect=study_round):
                    with self.assertRaises(RuntimeError) as raised:
                        self.run_session(pairs=self.pairs, device="cpu")
                self.assertIs(raised.exception, fault)
                studied = self.studied("second/model") if stage == "round" else []
                self.assertEqual(self.events, [PS, PS] + self.studied("first")
                                 + [PS, cpu_load("second/model")] + studied
                                 + [("unload", "second/model"), PS])
                self.assertEqual(run.loaded, user)
                report = json.loads(self.report_path.read_text())
                self.assertEqual((report["status"], report["failed"], report["error"]),
                                 ("failed", "second/model", repr(fault)))
                # An entry is appended once its source is calibrated, as before.
                self.assertEqual([e["name"] for e in report["sources"]],
                                 ["first"] + (["second/model"] if stage == "round" else []))

    def test_cpu_unload_failure_is_a_cleanup_error_and_the_source_error_wins(self):
        for raises in (True, False):
            with self.subTest(raises=raises):
                self.events.clear()
                fault = RuntimeError("round failed")
                user = [model("first", 128000, 4)]
                run = self.lms(user, fail=[("unload", "second/model")])

                def study_round(memory, stash, teacher, ledger, source, **kwargs):
                    self.event(("round", source))
                    if raises and source == "second/model":
                        raise fault
                    return {"used_up": True}

                with mock.patch.object(frontier, "study_round", side_effect=study_round):
                    with self.assertRaises(Exception) as raised:
                        self.run_session(pairs=self.pairs, device="cpu")
                report = json.loads(self.report_path.read_text())
                self.assertEqual((report["status"], report["failed"]), ("failed", "second/model"))
                if raises:
                    self.assertIs(raised.exception, fault)
                    self.assertTrue(report["error"].startswith(
                        repr(fault) + "; cleanup: CalledProcessError("), report["error"])
                    self.assertEqual(len(fault.__notes__), 1)
                else:
                    self.assertIsInstance(raised.exception, subprocess.CalledProcessError)
                    self.assertEqual(report["error"], repr(raised.exception))
                # The next source is never placed, and the restore unloads the leftover.
                self.assertEqual(self.events, [PS, PS] + self.studied("first")
                                 + [PS, cpu_load("second/model")] + self.studied("second/model")
                                 + [("unload", "second/model"), PS, ("unload", "second/model")])
                self.assertEqual(run.loaded, user)

    def test_report_device_and_seconds_span_placement_to_cleanup(self):
        pause = 0.05

        def load_alongside(name, context):
            time.sleep(pause)
            return "shared" if name == "first" else "cpu"
        self.swapper.load_alongside.side_effect = load_alongside
        self.swapper.unload.side_effect = lambda name: time.sleep(pause)
        self.swapper.load.side_effect = lambda name, context: time.sleep(pause)
        with mock.patch.object(frontier, "study_round", return_value={"used_up": True}):
            cpu = self.run_session(pairs=self.pairs, device="cpu")
            self.assertEqual(json.loads(self.report_path.read_text()), cpu)
            gpu = self.run_session(pairs=self.pairs, device="gpu")
        self.assertEqual([e["device"] for e in cpu["sources"]], ["shared", "cpu", "cpu"])
        self.assertEqual([e["device"] for e in gpu["sources"]], ["gpu"] * 3)
        self.assertEqual(self.swapper.unload.call_args_list,
                         [mock.call("second/model"), mock.call("third")])
        # Each span holds its placement, and its unload where there is one.
        for entry, pauses in zip(cpu["sources"] + gpu["sources"], (1, 2, 2, 1, 1, 1)):
            self.assertIsInstance(entry["seconds"], float)
            self.assertGreaterEqual(entry["seconds"], 0.9 * pause * pauses)

    def test_gpu_device_is_todays_path(self):
        with mock.patch.object(frontier, "study_round", side_effect=self.study_round):
            default = self.run_session(pairs=self.pairs)
            events = self.events[:]
            self.events.clear()
            explicit = self.run_session(pairs=self.pairs, device="gpu")
        self.assertEqual(self.events, events)
        self.assertEqual(events, ["snapshot"] + [event for source in self.plan for event in
                                                 [("load", source.name)] + self.studied(source.name)]
                         + [("restore", self.loaded)])
        self.swapper.load_alongside.assert_not_called()
        self.swapper.unload.assert_not_called()
        for report in (default, explicit):
            self.assertEqual([e["device"] for e in report["sources"]], ["gpu"] * 3)

    def test_bad_device_raises_before_the_backup(self):
        before = tree(self.home)
        with mock.patch.object(session, "backup", side_effect=AssertionError("backup")):
            for device in ("GPU", "cuda", "", None):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    self.run_session(pairs=self.pairs, device=device)
        self.assertEqual(self.swapper.mock_calls, [])
        self.assertEqual(tree(self.home), before)
        self.assertFalse(self.backups.exists())

    def test_cli_device_defaults_to_load_device_and_reaches_the_session(self):
        common = ["--root", str(self.root), "--backup-dir", str(self.backups),
                  "--report", str(self.report_path)]
        cases = (("cpu", [], "cpu"), ("gpu", [], "gpu"),
                 ("gpu", ["--device", "cpu"], "cpu"), ("cpu", ["--device", "gpu"], "gpu"))
        with mock.patch.object(session, "load_plan", return_value=self.plan), \
                mock.patch.object(session, "LMStudioSwapper", return_value=self.swapper), \
                mock.patch.object(session, "run_session", return_value={}) as run_session:
            for default, argv, device in cases:
                with self.subTest(default=default, argv=argv), \
                        mock.patch.object(session, "load_device", return_value=default) as load, \
                        mock.patch("sys.stdout", io.StringIO()):
                    self.assertEqual(session.main(common + argv), 0)
                    load.assert_called_with()
                    self.assertEqual(run_session.call_args.args, (self.root, self.plan))
                    self.assertEqual(run_session.call_args.kwargs["device"], device)
            with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                session.main(common + ["--device", "tpu"])
        self.assertEqual(run_session.call_count, len(cases))

    def test_dry_run_prints_the_device_and_writes_nothing(self):
        before = tree(self.home)
        for default, argv, device in (("cpu", ["--device", "gpu"], "gpu"),
                                      ("gpu", [], "gpu"), ("gpu", ["--device", "cpu"], "cpu")):
            with self.subTest(default=default, argv=argv):
                output = io.StringIO()
                with mock.patch.object(session, "load_plan", return_value=self.plan), \
                        mock.patch.object(session, "load_device", return_value=default), \
                        mock.patch.object(session, "LMStudioSwapper", return_value=self.swapper), \
                        mock.patch.object(session, "run_session", side_effect=AssertionError("run")), \
                        mock.patch.object(session, "backup", side_effect=AssertionError("backup")), \
                        mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")), \
                        mock.patch.object(Path, "write_text", side_effect=AssertionError("write")), \
                        mock.patch.object(SystematicMemory, "save", side_effect=AssertionError("save")), \
                        mock.patch("sys.stdout", output):
                    self.assertEqual(session.main(["--root", str(self.root), "--dry-run", *argv]), 0)
                self.assertEqual(json.loads(output.getvalue())["device"], device)
        self.assertEqual(tree(self.home), before)
        self.assertEqual([name for name, _, _ in self.swapper.mock_calls], ["snapshot"] * 3)


if __name__ == "__main__":
    unittest.main()
