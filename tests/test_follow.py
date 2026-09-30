"""§11.172: everything that holds the session follows the rebuild.

A take-in (§11.171) gives the window a new session built from the merged
library. The Learn tab's learner moves to it with its questions as they are,
and the actions that write the library on the UI thread wait for a running job
instead of writing beside it. The take-in is built with ``session.write_inbox``
on the synthetic staged library of tests/test_inbox.py. The window is built
withdrawn and pumped, as tests/test_gui.py does, and every test skips without a
display.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests.test_inbox import _hashes, _library, _value
from ultraquant import gui as gui_module
from ultraquant.distill import session
from ultraquant.interpreter.learning import LearningSession
from ultraquant.shards.vault import ShardVault

try:  # pragma: no cover - depends on the machine
    import tkinter as tk

    _root_probe = tk.Tk()
    _root_probe.destroy()
    _TK_OK = True
except Exception:  # noqa: BLE001 - headless or no tcl/tk
    _TK_OK = False

BUSY = "Busy - wait for the current task to finish."
PROBE = "follow-attach-probe"


class _Window:
    """A withdrawn window on the home that ``_home`` builds, semantic suggestions off."""

    def _home(self) -> Path:
        raise NotImplementedError

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_follow_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        previous = os.environ.get("ULTRAQUANT_CONFIG")
        settings = self.tmp / "settings.json"
        settings.write_text(json.dumps({"lmstudio": {"semantic_suggest": False},
                                        "stash_auto_approve": False}), encoding="utf-8")
        os.environ["ULTRAQUANT_CONFIG"] = str(settings)
        self.addCleanup(self._restore_config, previous)
        self.addCleanup(setattr, gui_module, "INBOX_SECONDS", gui_module.INBOX_SECONDS)
        self.home = self._home()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self._destroy)
        self.app = gui_module.UltraQuantGUI(self.root, self.home)
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


@unittest.skipUnless(_TK_OK, "Tk display not available")
class LearnerFollowsTests(_Window, unittest.TestCase):
    """The Learn tab's learner holds whatever session the window holds."""

    def _home(self):
        self.base, self.staging = _library(self.tmp, vault=True)
        live = self.tmp / "live" / "uq_home"
        shutil.copytree(self.base, live)
        return live

    def _settle(self, timeout: float = 120.0) -> bool:
        """Pump until no entry is pending and the worker is idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if not self.app.busy and not list((self.home / "inbox").glob("*.json")):
                return True
            time.sleep(0.02)
        return False

    def _episodes(self):
        return json.loads((self.home / "memory.json").read_text(encoding="utf-8"))["episodes"]

    def test_a_learner_surveyed_before_a_take_in_holds_the_new_session(self):
        gui_module.INBOX_SECONDS = 0
        self.app._survey()
        self.assertTrue(self._wait(timeout=120))
        learner, old = self.app.learner, self.app.session
        self.assertIs(learner.session, old)
        pending = list(learner.pending)
        self.assertTrue(pending, "the survey found nothing to ask")
        waiting = learner.next_question()

        path = session.write_inbox(self.home, self.staging, self.base, {"status": "complete"})
        self.assertTrue(self._settle())
        self.assertTrue((self.home / "inbox" / "applied" / path.name).exists())
        self.assertIsNot(self.app.session, old)
        self.assertEqual(self.app.session.memory.recall_fact("capital of mali")["value"], "Bamako")

        # The same learner, now on the new session, its questions as they were.
        self.assertIs(self.app.learner, learner)
        self.assertIs(learner.session, self.app.session)
        self.assertEqual(len(learner.pending), len(pending))
        for now, before in zip(learner.pending, pending):
            self.assertIs(now, before)
        self.assertIs(learner.next_question(), waiting)

        # Its next answer saves through the new session. The old one would
        # write the library as it was, and the merge's episodes would be lost.
        merged = self._episodes()
        self.app._answer_question("something")
        self.assertTrue(self._wait(timeout=120))
        self.assertIn(waiting.prompt, self.app.learn_log.get("1.0", "end"))
        episodes = self._episodes()
        self.assertEqual(episodes[:len(merged)], merged)
        self.assertEqual([episode["kind"] for episode in episodes[len(merged):]], ["learning"])
        self.assertEqual(_value(self.home, "capital of mali"), "Bamako")
        self.assertEqual(_value(self.home, "capital of chad"), "N'Djamena")

    def test_the_rebuild_itself_carries_the_learner(self):
        learner = LearningSession(self.app.session)
        pending = learner.survey()
        self.app.learner = learner
        old = self.app.session
        self.app._rebuild_session()
        self.assertIsNot(self.app.session, old)
        self.assertIs(learner.session, self.app.session)
        self.assertIs(learner.pending, pending)
        # No learner, or none yet: the rebuild goes ahead without one.
        self.app.learner = None
        self.app._rebuild_session()
        self.assertIsNone(self.app.learner)


@unittest.skipUnless(_TK_OK, "Tk display not available")
class WritersWaitTests(_Window, unittest.TestCase):
    """The UI-thread writers refuse while a job runs, and work once it is done."""

    def _home(self):
        return self.tmp / "home"

    def setUp(self):
        super().setUp()
        stash = self.app.session.stash

        def claim(site, subject, value):
            return stash.add_claim(
                f"https://{site}.example/{subject}", f"What is the capital of {subject}?",
                f"The capital of {subject} is {value}.", measured_confidence=0.9,
                fields={"key": f"capital of {subject}".lower(), "subject": subject,
                        "attribute": "capital", "value": value})
        # Two sources that disagree, so Analyze has something to write.
        self.rivals = [claim("one", "Testland", "Testville"), claim("two", "Testland", "Otherville")]
        self.promotable = claim("three", "Promoland", "Promoville")
        self.rejectable = claim("four", "Rejectland", "Rejectville")
        self.app.session.save()
        self.app._refresh_stash()

    def _hold(self):
        """Start a job that runs until the returned event is set."""
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def job():
            started.set()
            release.wait(120)
        self.app._run_async("Holding the worker", job)
        self.assertTrue(started.wait(10), "the job did not start")
        self.assertTrue(self.app.busy)
        return release

    def _select(self, entry_id):
        for item in self.app.stash_view.get_children():
            if int(self.app.stash_view.item(item, "values")[0]) == entry_id:
                self.app.stash_view.selection_set([item])
                return
        self.fail(f"entry {entry_id} is not in the stash table")

    def _files(self):
        """stash.json and memory.json byte for byte, and a digest of every file."""
        return ({name: (self.home / name).read_bytes() for name in ("stash.json", "memory.json")},
                _hashes(self.home))

    def _statuses(self):
        return {entry["id"]: entry["status"] for entry in self.app.session.stash.entries()}

    def test_promote_reject_and_analyze_wait_for_a_running_job_then_work(self):
        release = self._hold()
        before = self._files()
        statuses = self._statuses()
        for name, entry_id, action in (
            ("promote", self.promotable, self.app._promote),
            ("promote (force)", self.promotable, lambda: self.app._promote(force=True)),
            ("reject", self.rejectable, self.app._reject),
            ("analyze", None, self.app._analyze),
        ):
            with self.subTest(action=name):
                if entry_id is not None:
                    self._select(entry_id)
                self.app.status.set("")
                action()
                self.root.update()
                self.assertEqual(self.app.status.get(), BUSY)
                self.assertEqual(self._files(), before)
                self.assertEqual(self._statuses(), statuses)
        self.assertTrue(self.app.busy)

        release.set()
        self.assertTrue(self._wait())
        self.assertEqual(self.app.status.get(), "ready")

        stash, memory = before[0]["stash.json"], before[0]["memory.json"]
        self._select(self.promotable)
        self.app._promote()
        self.assertEqual(self.app.status.get(), "ready")
        self.assertNotEqual((self.home / "stash.json").read_bytes(), stash)
        self.assertNotEqual((self.home / "memory.json").read_bytes(), memory)
        self.assertEqual(self._statuses()[self.promotable], "promoted")
        self.assertEqual(self.app.session.memory.recall_fact("capital of promoland")["value"],
                         "Promoville")
        self.assertIn(f"[stash] promoted {self.promotable} to fact 'capital of promoland'",
                      self.app.transcript.get("1.0", "end"))

        stash = (self.home / "stash.json").read_bytes()
        self._select(self.rejectable)
        self.app._reject()
        self.assertEqual(self.app.status.get(), "ready")
        self.assertNotEqual((self.home / "stash.json").read_bytes(), stash)
        self.assertEqual(self._statuses()[self.rejectable], "rejected")

        stash = (self.home / "stash.json").read_bytes()
        self.app._analyze()
        self.assertEqual(self.app.status.get(), "ready")
        self.assertNotEqual((self.home / "stash.json").read_bytes(), stash)
        self.assertEqual([self._statuses()[entry_id] for entry_id in self.rivals],
                         ["disputed", "disputed"])

    def test_attach_waits_without_opening_its_dialog(self):
        elsewhere = ShardVault(self.tmp / "elsewhere" / "vault")
        elsewhere.add_shard(PROBE, "testing", {"note": "attached by tests/test_follow.py"},
                            kind="note")
        library = self.tmp / "elsewhere" / "probe.uql"
        self.assertEqual(elsewhere.pack(library), 1)

        release = self._hold()
        before = self._files()
        self.app.status.set("")
        with mock.patch.object(gui_module.filedialog, "askopenfilename",
                               return_value=str(library)) as dialog:
            self.app._attach()
            self.root.update()
        dialog.assert_not_called()
        self.assertEqual(self.app.status.get(), BUSY)
        self.assertEqual(self._files(), before)
        self.assertFalse(self.app.session.vault.has(PROBE))

        release.set()
        self.assertTrue(self._wait())
        vault = _hashes(self.home / "vault")
        with mock.patch.object(gui_module.filedialog, "askopenfilename",
                               return_value=str(library)) as dialog:
            self.app._attach()
        dialog.assert_called_once()
        self.assertTrue(self.app.session.vault.has(PROBE))
        self.assertNotEqual(_hashes(self.home / "vault"), vault)
        self.assertIn(f"[library] attached 1 shard(s) from {library}",
                      self.app.transcript.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
