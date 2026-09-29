"""§11.171: a study session announces itself in the inbox, and the GUI takes it in.

The merge (ultraquant.distill.merge), the announcement and ``--stage``
(ultraquant.distill.session) run on synthetic libraries built the way
tests/test_merge_session.py builds them, with no LM Studio, WordNet or real
library. The GUI tests build the window withdrawn and pump it, as
tests/test_gui.py does, and skip without a display.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time
import unittest
from unittest import mock

from ultraquant import gui as gui_module
from ultraquant.distill import frontier, merge as M, session, sources
from ultraquant.distill.teachers import TeacherSpec
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.systematic import SystematicMemory

try:  # pragma: no cover - depends on the machine
    import tkinter as tk

    _root_probe = tk.Tk()
    _root_probe.destroy()
    _TK_OK = True
except Exception:  # noqa: BLE001 - headless or no tcl/tk
    _TK_OK = False

ROOT = Path(__file__).resolve().parents[1]
GUI_CLAIM = "The capital of Testland is Testville."
PRE_MERGE = r"^uq_home-\d{8}T\d{6}Z-pre-merge$"


def _claim(stash, run_id, subject, attribute, value, teachers=("first",), **extra):
    question_id = f"{attribute}:{subject}".lower()
    return stash.add_claim(
        f"https://distill.invalid/{run_id}/{question_id}", f"What is the {attribute} of {subject}?",
        f"The {attribute} of {subject} is {value}.", measured_confidence=0.9,
        provenance={"run_id": run_id, "question_id": question_id, "teachers": list(teachers),
                    "lineages": list(teachers), **extra},
        fields={"key": f"{attribute} of {subject}".lower(), "subject": subject,
                "attribute": attribute, "value": value})


def _hashes(root, skip=()):
    """Every file's digest, leaving out the top-level folders in ``skip``."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] not in skip}


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _library(tmp, vault=False):
    """A base library, and a session staged on a copy of it, as tests/test_merge_session.py has them.

    ``vault`` keeps the facts in a shard vault, as the GUI's own library does.
    """
    base = tmp / "base" / "uq_home"
    base.mkdir(parents=True)
    if vault:
        (base / "vault").mkdir()
    memory, stash, approver = M._open(base)
    seeded = _claim(stash, "seed-1", "France", "capital", "Paris")
    _claim(stash, "seed-1", "Spain", "capital", "Madrid")
    approver.approve_all()
    memory.save()
    (base / "sources.json").write_text(json.dumps({"first": [["q0", True]]}), encoding="utf-8")
    # The session: two rounds of one source, then a second source that
    # corroborates a seeded claim and revises a first-round one.
    staging = tmp / "staging" / "uq_home"
    shutil.copytree(base, staging)
    memory, stash, approver = M._open(staging)
    _claim(stash, "session-T-first-1", "Chad", "capital", "N'Djamena")
    approver.approve_all()
    _claim(stash, "session-T-first-2", "Mali", "capital", "Timbuktu")
    approver.approve_all()
    stash.add_teacher(seeded, "second", "second.gguf:2", prior_teacher_ids=["first"])
    _claim(stash, "session-T-second-1", "Mali", "capital", "Bamako", teachers=("second", "third"),
           settles="capital of mali")
    approver.approve_all()
    memory.learn_kind("capital", "country")
    memory.learn_kind("population", None)
    memory.save()
    (staging / "sources.json").write_text(json.dumps(
        {"first": [["q0", True], ["q1", True], ["q2", True]], "second": [["q3", True]]}),
        encoding="utf-8")
    return base, staging


def _value(root, key):
    memory, _, _ = M._open(root)
    return (memory.recall_fact(key) or {}).get("value")


def _ledger(root):
    return json.loads((root / "sources.json").read_text(encoding="utf-8"))


class WriteInboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_inbox_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_entry_is_named_by_the_stamp_and_names_both_libraries(self):
        home = self.tmp / "home"
        staging = self.tmp / "stage" / "uq_home"
        base = self.tmp / "stage" / "backups" / ".." / "backups" / "uq_home-20260929T000000000000Z"
        report = {"status": "complete", "backup": str(base), "sources": []}
        with mock.patch.object(session, "_stamp", return_value="20260929T120000000001Z"):
            path = session.write_inbox(home, staging, base, report)
        self.assertEqual(path, home / "inbox" / "20260929T120000000001Z.json")
        # Nothing half-written is left beside it for the GUI to find.
        self.assertEqual(list((home / "inbox").iterdir()), [path])
        entry = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(entry), {"staging", "base", "report", "created"})
        self.assertEqual(entry["staging"], str(staging.resolve()))
        self.assertEqual(entry["base"], str(base.resolve()))
        self.assertNotIn("..", entry["base"])
        self.assertEqual(entry["report"], report)
        created = datetime.fromisoformat(entry["created"])
        self.assertEqual(created.utcoffset(), timedelta(0))
        self.assertLess(abs(datetime.now(timezone.utc) - created), timedelta(minutes=5))

    def test_names_are_utc_stamps_that_sort_in_the_order_written(self):
        home = self.tmp / "home"
        first = session.write_inbox(home, self.tmp, self.tmp, {})
        time.sleep(0.002)
        second = session.write_inbox(home, self.tmp, self.tmp, {})
        for path in (first, second):
            self.assertRegex(path.name, r"^\d{8}T\d{12}Z\.json$")
        self.assertLess(first.name, second.name)
        self.assertEqual(sorted(p.name for p in (home / "inbox").iterdir()), [first.name, second.name])


class MergeModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_inbox_merge_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.base, self.staging = _library(self.tmp)
        self.backups = self.tmp / "b"

    def _live(self, name, gui=False):
        live = self.tmp / name / "uq_home"
        shutil.copytree(self.base, live)
        if gui:
            memory, stash, _ = M._open(live)
            memory.remember_fact("capital of spain", "Madrid (corrected in the GUI)", 0.99,
                                 subject="Spain", attribute="capital")
            memory.save()
            stash.add_claim("https://example.org/lookup", "A GUI lookup", GUI_CLAIM,
                            measured_confidence=0.9)
        return live

    def test_the_tool_is_the_package_command_line(self):
        spec = importlib.util.spec_from_file_location("merge_session_tool",
                                                      ROOT / "tools" / "merge_session.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        self.assertIs(tool.main, M.main)

    def test_replay_true_replays_into_an_unchanged_library(self):
        live = self._live("unchanged")
        self.assertEqual(_hashes(live), _hashes(self.base))
        with mock.patch.object(M, "_replace", side_effect=AssertionError("replaced")):
            report = M.merge(self.staging, live, self.base, replay=True)
        self.assertTrue(report["clean"], report)
        self.assertIn("replayed", report["mode"])
        self.assertEqual((report["replayed"], report["batches"], report["hidden"]), (3, 3, 0))
        self.assertEqual(_value(live, "capital of chad"), "N'Djamena")
        self.assertEqual(_value(live, "capital of mali"), "Bamako")
        self.assertEqual(_ledger(live), _ledger(self.staging))

    def test_replay_is_looked_up_on_the_module_when_merge_asks_for_it(self):
        live = self._live("patched")
        real, calls = M.replay, []

        def spy(staging, live_, base):
            calls.append(Path(live_))
            return real(staging, live_, base)
        with mock.patch.object(M, "replay", spy):
            report = M.merge(self.staging, live, self.base, replay=True)
        self.assertEqual(calls, [live])
        self.assertTrue(report["clean"], report)

    def test_a_clean_check_backs_up_then_merges(self):
        for replay in (False, True):
            with self.subTest(replay=replay):
                live = self._live(f"clean-{replay}", gui=True)
                backups = self.backups / str(replay)
                before = _hashes(live)
                real, seen = M.merge, []

                def spy(staging, into, base, *, replay=False):
                    # Each merge: into the library or a scratch copy, and the backups made by then.
                    seen.append((Path(into) == live, replay,
                                 [_hashes(p) for p in backups.glob("*")] if backups.exists() else []))
                    return real(staging, into, base, replay=replay)
                with mock.patch.object(M, "merge", spy):
                    report = M.checked_merge(self.staging, live, self.base, backups, replay=replay)
                self.assertTrue(report["clean"], report)
                self.assertEqual(seen, [(False, replay, []), (True, replay, [before])])
                backup = Path(report["backup"])
                self.assertEqual(backup.parent, backups)
                self.assertRegex(backup.name, PRE_MERGE)
                self.assertEqual(_hashes(backup), before)
                self.assertNotIn("check", report)  # the report of the merge into the library
                self.assertEqual(_value(live, "capital of mali"), "Bamako")
                self.assertEqual(_value(live, "capital of spain"), "Madrid (corrected in the GUI)")

    def test_an_unclean_check_writes_nothing(self):
        live = self._live("unclean", gui=True)
        before = _hashes(live)
        with mock.patch.object(M, "_as_the_session_saw_it",
                               lambda stash, hidden: contextlib.nullcontext()):
            report = M.checked_merge(self.staging, live, self.base, self.backups, replay=True)
        self.assertFalse(report["clean"])
        self.assertEqual(len(report["touched"]), 1)
        self.assertEqual(report["check"], "on a scratch copy; the live library was not written")
        self.assertNotIn("backup", report)
        self.assertEqual(_hashes(live), before)
        self.assertFalse(self.backups.exists())

    def test_a_ledger_that_moved_on_raises_and_writes_nothing(self):
        live = self._live("moved")
        (live / "sources.json").write_text(json.dumps({"first": [["q0", True], ["other", False]]}),
                                           encoding="utf-8")
        before = _hashes(live)
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            M.checked_merge(self.staging, live, self.base, self.backups, replay=True)
        self.assertEqual(_hashes(live), before)
        self.assertFalse(self.backups.exists())


class FakeTeacher:
    """Stands in for sources.LMStudioTeacher: every sample is the held answer."""

    def __init__(self, name, gguf):
        self.spec = TeacherSpec(name, Path(gguf))

    def ask(self, questions, *, samples, **kwargs):
        return [["Alpha"] * samples for _ in questions]


class StageTests(unittest.TestCase):
    """``--stage``: the session studies a copy and announces itself; fakes throughout."""

    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.home / "library"
        self.root.mkdir()
        gguf = self.home / "fake.gguf"
        name, value = b"general.name", b"Fake teacher"
        gguf.write_bytes(struct.pack("<4sIQQQ", b"GGUF", 3, 0, 1, len(name))
                         + name + struct.pack("<IQ", 8, len(value)) + value)
        self.plan = [session.SourcePlan(n, str(gguf), 4096) for n in ("first", "second/model")]
        memory = SystematicMemory(self.root / "memory.json")
        memory.remember_fact("capital of somewhere", "Alpha", 0.99,
                             subject="Somewhere", attribute="capital")
        memory.save()
        stash = ContemporaryStash(self.root / "stash.json")
        stash.add_claim("https://distill.invalid/test/capital:Somewhere",
                        "What is the capital of Somewhere?", "The capital of Somewhere is Alpha.",
                        provenance={"teachers": [s.name for s in self.plan],
                                    "run_id": "test", "question_id": "capital:Somewhere"},
                        fields={"key": "capital of somewhere", "subject": "Somewhere",
                                "attribute": "capital", "value": "Alpha"})
        stash._entries[1]["status"] = "promoted"
        stash.save()
        self.loaded = [dict(identifier="original", type="llm", contextLength=8192,
                            parallel=4, ttlMs=3600000)]
        self.swapper = mock.Mock()
        self.swapper.snapshot.side_effect = lambda: deepcopy(self.loaded)
        self.swapper.load_alongside.return_value = "shared"
        # Unexpected external calls fail instead of reaching a process or network.
        self.enterContext(mock.patch("subprocess.run", side_effect=AssertionError("external process")))
        self.enterContext(mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")))

    @staticmethod
    def study_round(memory, stash, teacher, ledger, source, **kwargs):
        memory.remember_fact(f"studied by {source}", "yes", 0.9)
        return {"used_up": True}

    def main(self, *argv, study_round=None):
        output = io.StringIO()
        with mock.patch.object(session, "load_plan", return_value=self.plan), \
                mock.patch.object(session, "LMStudioSwapper", return_value=self.swapper), \
                mock.patch.object(sources, "LMStudioTeacher", FakeTeacher), \
                mock.patch.object(frontier, "study_round", side_effect=study_round or self.study_round), \
                mock.patch("sys.stdout", output):
            code = session.main(["--root", str(self.root), *argv])
        return code, output.getvalue()

    def staged(self):
        found = list((self.home.resolve() / "uq_backups").glob("staging-*"))
        self.assertEqual(len(found), 1, found)
        self.assertRegex(found[0].name, r"^staging-\d{8}-\d{6}$")
        return found[0]

    def test_stage_studies_a_copy_and_announces_it_in_the_inbox(self):
        before = tree(self.root)
        code, output = self.main("--stage")
        self.assertEqual(code, 0)
        staged = self.staged()
        # The library is untouched but for its inbox; the copy did the studying.
        self.assertEqual({k: v for k, v in tree(self.root).items()
                          if Path(k).parts[0] != "inbox"}, before)
        copy = staged / "uq_home"
        self.assertEqual(SystematicMemory(copy / "memory.json").recall_fact("studied by first")["value"],
                         "yes")
        self.assertIsNone(SystematicMemory(self.root / "memory.json").recall_fact("studied by first"))
        # The report and the base, beside the copy, where tools/merge_session.py looks.
        report = json.loads((staged / "session-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "complete")
        base = Path(report["backup"])
        self.assertEqual(base.parent, staged / "backups")
        self.assertEqual(tree(base), before)
        self.assertEqual(M.default_base(copy), base)
        # One entry, naming both, and its path printed.
        entries = list((self.root / "inbox").iterdir())
        self.assertEqual(len(entries), 1)
        self.assertRegex(entries[0].name, r"^\d{8}T\d{12}Z\.json$")
        entry = json.loads(entries[0].read_text(encoding="utf-8"))
        self.assertEqual(entry["staging"], str(copy.resolve()))
        self.assertEqual(entry["base"], str(base.resolve()))
        self.assertEqual(entry["report"], report)
        self.assertEqual(output.splitlines()[-1], str(entries[0]))

    def test_explicit_backup_dir_and_report_win(self):
        mine = self.home / "mine"
        code, _ = self.main("--stage", "--backup-dir", str(mine / "backups"),
                            "--report", str(mine / "report.json"))
        self.assertEqual(code, 0)
        staged = self.staged()
        self.assertTrue((staged / "uq_home" / "memory.json").exists())
        self.assertFalse((staged / "backups").exists())
        self.assertFalse((staged / "session-report.json").exists())
        report = json.loads((mine / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(report["backup"]).parent, (mine / "backups").resolve())
        entry = json.loads(next((self.root / "inbox").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(entry["base"], str(Path(report["backup"]).resolve()))

    def test_a_failed_session_announces_nothing(self):
        fault = RuntimeError("round failed")

        def failing(memory, stash, teacher, ledger, source, **kwargs):
            raise fault
        with self.assertRaises(RuntimeError) as raised:
            self.main("--stage", study_round=failing)
        self.assertIs(raised.exception, fault)
        report = json.loads((self.staged() / "session-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertFalse((self.root / "inbox").exists())

    def test_a_report_that_is_not_complete_announces_nothing(self):
        with mock.patch.object(session, "run_session",
                               return_value={"status": "failed", "backup": str(self.home)}):
            code, output = self.main("--stage")
        self.assertEqual(code, 0)
        self.assertFalse((self.root / "inbox").exists())
        self.assertNotIn("inbox", output)

    def test_without_stage_nothing_is_copied_or_announced(self):
        code, _ = self.main("--backup-dir", str(self.home / "backups"),
                            "--report", str(self.home / "report.json"))
        self.assertEqual(code, 0)
        self.assertEqual(SystematicMemory(self.root / "memory.json").recall_fact(
            "studied by first")["value"], "yes")
        self.assertFalse((self.home / "uq_backups").exists())
        self.assertFalse((self.root / "inbox").exists())

    def test_stage_with_dry_run_is_refused(self):
        errors = io.StringIO()
        with mock.patch.object(session, "load_plan", side_effect=AssertionError("plan")), \
                mock.patch.object(session, "LMStudioSwapper", side_effect=AssertionError("swapper")), \
                mock.patch("sys.stderr", errors), self.assertRaises(SystemExit) as raised:
            session.main(["--root", str(self.root), "--stage", "--dry-run"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--stage and --dry-run cannot be combined", errors.getvalue())
        self.assertFalse((self.home / "uq_backups").exists())
        self.assertFalse((self.root / "inbox").exists())


@unittest.skipUnless(_TK_OK, "Tk display not available")
class GUIInboxTests(unittest.TestCase):
    """The window takes a finished session in itself, between its own tasks."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_gui_inbox_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        previous = os.environ.get("ULTRAQUANT_CONFIG")
        settings = self.tmp / "settings.json"
        settings.write_text(json.dumps({"lmstudio": {"semantic_suggest": False},
                                        "stash_auto_approve": False}), encoding="utf-8")
        os.environ["ULTRAQUANT_CONFIG"] = str(settings)
        self.addCleanup(self._restore_config, previous)
        self.addCleanup(setattr, gui_module, "INBOX_SECONDS", gui_module.INBOX_SECONDS)
        self.base, self.staging = _library(self.tmp, vault=True)
        self.live = self.tmp / "live" / "uq_home"
        shutil.copytree(self.base, self.live)
        self.inbox = self.live / "inbox"
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self._destroy)
        self.app = gui_module.UltraQuantGUI(self.root, self.live)
        self.assertTrue(self._wait(), "session did not start")

    @staticmethod
    def _restore_config(previous):
        if previous is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = previous

    def _destroy(self):
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - already gone
            pass

    def _wait(self, timeout: float = 60.0) -> bool:
        """Pump the event loop until the worker is idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if not self.app.busy:
                return True
            time.sleep(0.02)
        return False

    def _settle(self, timeout: float = 120.0) -> bool:
        """Pump until no entry is pending and the worker is idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if not self.app.busy and not list(self.inbox.glob("*.json")):
                return True
            time.sleep(0.02)
        return False

    def _announce(self):
        path = session.write_inbox(self.live, self.staging, self.base, {"status": "complete"})
        return path, json.loads(path.read_text(encoding="utf-8"))

    def _moved(self, folder, path):
        entry = json.loads((self.inbox / folder / path.name).read_text(encoding="utf-8"))
        return entry, entry.pop("outcome")

    def test_a_pending_session_is_taken_in_when_idle(self):
        gui_module.INBOX_SECONDS = 0
        memory = self.app.session.memory
        known = set(memory.fact_keys())
        self.app._submit("the tower height is 324 metres")
        self.assertTrue(self._wait())
        told = {key: memory.recall_fact(key)["value"] for key in set(memory.fact_keys()) - known}
        self.assertTrue(told, "the chat turn stored no fact")
        # Held in memory only: a turn saves as it ends, so this is what the
        # take-in's own save is for.
        memory.remember_fact("favourite colour of the tester", "blue", 0.9,
                             subject="the tester", attribute="favourite colour")
        told["favourite colour of the tester"] = "blue"
        old, old_cli = self.app.session, self.app.cli
        path, written = self._announce()
        with mock.patch.object(M, "checked_merge", wraps=M.checked_merge) as checked:
            self.assertTrue(self._settle())
        checked.assert_called_once_with(Path(written["staging"]), self.live, Path(written["base"]),
                                        self.live.parent / "uq_backups", replay=True)

        self.assertFalse(path.exists())
        self.assertFalse((self.inbox / "held").exists())
        entry, outcome = self._moved("applied", path)
        self.assertEqual(entry, written)
        self.assertEqual((outcome["clean"], outcome["error"]), (True, None))
        report = outcome["report"]
        self.assertTrue(report["clean"], report)
        self.assertIn("replayed", report["mode"])
        backup = Path(report["backup"])
        self.assertEqual(backup.parent, self.live.parent / "uq_backups")
        self.assertRegex(backup.name, PRE_MERGE)
        self.assertEqual(_value(backup, "capital of mali"), None)
        # A new session, built from the merged library: the session's
        # knowledge, and the fact the chat stored before it.
        self.assertIsNot(self.app.session, old)
        self.assertIsNot(self.app.cli, old_cli)
        self.assertIs(self.app.cli.session, self.app.session)
        memory = self.app.session.memory
        self.assertEqual(memory.recall_fact("capital of mali")["value"], "Bamako")
        self.assertEqual(memory.recall_fact("capital of chad")["value"], "N'Djamena")
        for key, value in told.items():
            self.assertEqual(memory.recall_fact(key)["value"], value)
        self.assertEqual(self.app.status.get(), "ready")
        self.assertIn(f"[study] took in {path.name}: {report['session changed']} changes; "
                      f"backup {report['backup']}", self.app.transcript.get("1.0", "end"))
        self.assertEqual(len(self.app.stash_view.get_children()), len(self.app.session.stash.entries()))

        # The next turn saves through the new session, keeping the merge.
        self.app._submit("the bridge length is 50 metres")
        self.assertTrue(self._wait())
        self.assertEqual(_value(self.live, "capital of mali"), "Bamako")
        for key, value in told.items():
            self.assertEqual(_value(self.live, key), value)
        self.assertEqual(_ledger(self.live), _ledger(self.staging))

    def test_an_unclean_session_is_held_and_nothing_is_written(self):
        gui_module.INBOX_SECONDS = 0
        # A claim staged here since the copy: the session's approvals must not
        # see it, and a replay that lets them is not clean.
        self.app.session.stash.add_claim("https://example.org/lookup", "A GUI lookup", GUI_CLAIM,
                                         measured_confidence=0.9)
        self.app.session.save()
        before = _hashes(self.live, skip={"inbox"})
        old = self.app.session
        path, written = self._announce()
        with mock.patch.object(M, "_as_the_session_saw_it",
                               lambda stash, hidden: contextlib.nullcontext()):
            self.assertTrue(self._settle())

        entry, outcome = self._moved("held", path)
        self.assertEqual(entry, written)
        self.assertEqual((outcome["clean"], outcome["error"]), (False, None))
        self.assertFalse(outcome["report"]["clean"])
        self.assertEqual(len(outcome["report"]["touched"]), 1)
        self.assertNotIn("backup", outcome["report"])
        self.assertFalse((self.inbox / "applied").exists())
        self.assertFalse((self.live.parent / "uq_backups").exists())
        self.assertEqual(_hashes(self.live, skip={"inbox"}), before)
        self.assertIs(self.app.session, old)
        self.assertIsNone(self.app.session.memory.recall_fact("capital of mali"))
        self.assertIn(f"[study] held {path.name}: ", self.app.transcript.get("1.0", "end"))

    def test_a_merge_that_raises_is_held_with_its_error(self):
        gui_module.INBOX_SECONDS = 0
        (self.live / "sources.json").write_text(json.dumps({"first": [["q0", True], ["other", False]]}),
                                                encoding="utf-8")
        old = self.app.session
        path, written = self._announce()
        self.assertTrue(self._settle())

        entry, outcome = self._moved("held", path)
        self.assertEqual(entry, written)
        self.assertEqual((outcome["clean"], outcome["report"]), (False, None))
        self.assertIn("ValueError", outcome["error"])
        self.assertIn("not a prefix", outcome["error"])
        self.assertFalse((self.live.parent / "uq_backups").exists())
        self.assertIs(self.app.session, old)
        self.assertIn(f"[study] held {path.name}: {outcome['error']}",
                      self.app.transcript.get("1.0", "end"))
        self.assertEqual(self.app.status.get(), "ready")

    def test_nothing_is_taken_in_while_busy(self):
        gui_module.INBOX_SECONDS = 0
        old = self.app.session
        self.app.busy = True
        try:
            path, _ = self._announce()
            for _ in range(10):
                self.app._pump_once()
                self.root.update()
                time.sleep(0.02)
            self.assertTrue(path.exists())
            self.assertFalse((self.inbox / "applied").exists())
            self.assertIs(self.app.session, old)
            # Not even asked: a refused start would say "Busy" on every look.
            self.assertNotIn("Busy", self.app.transcript.get("1.0", "end"))
        finally:
            self.app.busy = False
        self.assertTrue(self._settle())
        self.assertTrue((self.inbox / "applied" / path.name).exists())

    def test_the_inbox_is_looked_in_every_INBOX_SECONDS_read_at_each_look(self):
        gui_module.INBOX_SECONDS = 3600
        self.app._inbox_looked = None
        self.app._check_inbox()  # the first look runs, and finds nothing
        path, _ = self._announce()
        for _ in range(5):
            self.app._check_inbox()
            self.root.update()
        self.assertFalse(self.app.busy)
        self.assertTrue(path.exists())
        gui_module.INBOX_SECONDS = 0  # read at each look: the very next one runs
        self.app._check_inbox()
        self.assertTrue(self.app.busy)
        self.assertTrue(self._settle())
        self.assertTrue((self.inbox / "applied" / path.name).exists())

    def test_the_entry_whose_name_sorts_first_is_taken_in_alone(self):
        gui_module.INBOX_SECONDS = 3600
        # A folder that sorts first is not an entry: only files are.
        (self.inbox / "00-folder.json").mkdir(parents=True)
        second = session.write_inbox(self.live, self.staging, self.base, {})
        first = second.with_name("0" + second.name)
        shutil.copyfile(second, first)
        self.app._inbox_looked = None
        self.app._check_inbox()
        self.assertTrue(self.app.busy)
        self.assertTrue(self._wait(timeout=120))
        self.assertTrue((self.inbox / "applied" / first.name).exists())
        self.assertTrue(second.exists())
        self.assertTrue((self.inbox / "00-folder.json").is_dir())


if __name__ == "__main__":
    unittest.main()
