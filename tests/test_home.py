"""§11.173: the session folder stays put while the GUI works.

The Storage tab's "Open session here" (``_reopen_session``) and the "Change
session folder..." menu (``_choose_home``) refuse while a job runs, before they
name another folder and before the menu's dialog opens. A take-in (§11.171)
reads the window's folder again after its merge, to rebuild the session and to
file its entry, so a refused switch must leave that folder as it was. The
libraries are the synthetic ones of tests/test_inbox.py, the take-in is
announced with ``session.write_inbox`` as there, and the window is built
withdrawn and pumped, as tests/test_follow.py builds it. Every test skips
without a display.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
import time
import unittest
from unittest import mock

from tests.test_follow import BUSY, _TK_OK, _Window
from tests.test_inbox import _hashes, _library, _value
from ultraquant import gui as gui_module
from ultraquant.distill import merge as M, session


@unittest.skipUnless(_TK_OK, "Tk display not available")
class HomeStaysTests(_Window, unittest.TestCase):
    """Neither way of moving the window to another session folder runs beside a job."""

    def _home(self):
        self.base, self.staging = _library(self.tmp, vault=True)
        # Another session folder, one that knows what the session learned.
        self.elsewhere = self.tmp / "elsewhere" / "uq_home"
        shutil.copytree(self.staging, self.elsewhere)
        live = self.tmp / "live" / "uq_home"
        shutil.copytree(self.base, live)
        return live

    def _until(self, done, timeout: float = 60.0) -> bool:
        """Pump the event loop until ``done()`` holds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if done():
                return True
            time.sleep(0.02)
        return False

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

    def _switch(self, how, folder):
        """Ask the window to move to ``folder``, and return the folder dialog.

        "here" is the Storage tab's "Open session here", with the library
        directory at ``folder``'s vault; "menu" is "Change session folder...",
        with its dialog answering ``folder``.
        """
        with mock.patch.object(gui_module.filedialog, "askdirectory",
                               return_value=str(folder)) as dialog:
            if how == "here":
                self.app.library_root.set(str(folder / "vault"))
                self.app._reopen_session()
            else:
                self.app._choose_home()
            self.root.update()
        return dialog

    def _refused(self, how):
        """A move asked for while a job runs says Busy, and changes nothing."""
        home, old = self.app.home, self.app.session
        files = _hashes(self.home), _hashes(self.elsewhere)
        self.app.status.set("")
        dialog = self._switch(how, self.elsewhere)
        dialog.assert_not_called()
        self.assertEqual(self.app.status.get(), BUSY)
        self.assertEqual(self.app.home, home)
        self.assertIs(self.app.session, old)
        self.assertEqual((_hashes(self.home), _hashes(self.elsewhere)), files)
        # Refused before anything else: not even the line announcing the move.
        for pane in (self.app.transcript, self.app.storage_log):
            self.assertNotIn(str(self.elsewhere), pane.get("1.0", "end"))

    def _waits_then_moves(self, how):
        old = self.app.session
        release = self._hold()
        self._refused(how)
        self.assertTrue(self.app.busy)

        release.set()
        self.assertTrue(self._wait())
        # Refused, not put off: with the job done, the window is where it was.
        self.assertEqual(self.app.status.get(), "ready")
        self.assertEqual(self.app.home, self.home)
        self.assertIs(self.app.session, old)
        self.assertIsNone(old.memory.recall_fact("capital of mali"))

        dialog = self._switch(how, self.elsewhere)
        self.assertEqual(dialog.call_count, 1 if how == "menu" else 0)
        self.assertTrue(self._wait())
        self.assertEqual(self.app.status.get(), "ready")
        self.assertEqual(self.app.home, self.elsewhere)
        self.assertIsNot(self.app.session, old)
        self.assertEqual(self.app.session.root, self.elsewhere)
        self.assertIs(self.app.cli.session, self.app.session)
        self.assertEqual(self.app.session.memory.recall_fact("capital of mali")["value"], "Bamako")
        self.assertEqual(self.app.library_root.get(), str(self.elsewhere / "vault"))
        if how == "here":
            self.assertIn(f"\nReopening session at {self.elsewhere}\n",
                          self.app.storage_log.get("1.0", "end"))
        else:
            self.assertIn(f"\n[session] switching to {self.elsewhere}\n",
                          self.app.transcript.get("1.0", "end"))
        self.assertIn(f"UltraQuant session at {self.elsewhere}\n",
                      self.app.transcript.get("1.0", "end"))

    def test_open_session_here_waits_for_a_running_job_then_moves(self):
        self._waits_then_moves("here")

    def test_change_session_folder_waits_without_opening_its_dialog_then_moves(self):
        self._waits_then_moves("menu")

    def test_a_take_in_rebuilds_and_files_its_entry_in_the_folder_it_began_in(self):
        gui_module.INBOX_SECONDS = 0
        old = self.app.session
        elsewhere = _hashes(self.elsewhere)
        real, started, release = M.checked_merge, threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def held(*args, **kwargs):
            started.set()
            release.wait(120)
            return real(*args, **kwargs)
        with mock.patch.object(M, "checked_merge", side_effect=held) as checked:
            path = session.write_inbox(self.home, self.staging, self.base, {"status": "complete"})
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(self._until(started.is_set), "the take-in did not start")
            self.assertTrue(self.app.busy)
            for how in ("here", "menu"):
                with self.subTest(action=how):
                    self._refused(how)
            release.set()
            self.assertTrue(self._until(
                lambda: not self.app.busy and not list((self.home / "inbox").glob("*.json")),
                timeout=120))
        checked.assert_called_once_with(Path(written["staging"]), self.home, Path(written["base"]),
                                        self.home.parent / "uq_backups", replay=True)

        # Filed in the folder the take-in began in, and nothing written elsewhere.
        applied = self.home / "inbox" / "applied" / path.name
        self.assertTrue(applied.exists())
        self.assertIs(json.loads(applied.read_text(encoding="utf-8"))["outcome"]["clean"], True)
        self.assertFalse(path.exists())
        self.assertFalse((self.home / "inbox" / "held").exists())
        self.assertFalse((self.elsewhere / "inbox").exists())
        self.assertEqual(_hashes(self.elsewhere), elsewhere)
        self.assertEqual(self.app.home, self.home)
        # Rebuilt from the library the merge went into.
        self.assertIsNot(self.app.session, old)
        self.assertEqual(self.app.session.root, self.home)
        self.assertIs(self.app.cli.session, self.app.session)
        self.assertEqual(self.app.session.memory.recall_fact("capital of mali")["value"], "Bamako")
        self.assertEqual(_value(self.home, "capital of mali"), "Bamako")
        self.assertIn(f"[study] took in {path.name}: ", self.app.transcript.get("1.0", "end"))
        self.assertEqual(self.app.status.get(), "ready")


if __name__ == "__main__":
    unittest.main()
