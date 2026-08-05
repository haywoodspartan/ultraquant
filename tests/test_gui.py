"""Tests for the desktop front end.

Tk needs a display, so every test here skips cleanly on a headless machine.
The window is built withdrawn (never mapped on screen) and driven by pumping
``update()`` rather than entering ``mainloop()``, so the suite stays automatic.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from ultraquant import gui as gui_module

try:  # pragma: no cover - depends on the machine
    import tkinter as tk

    _root_probe = tk.Tk()
    _root_probe.destroy()
    _TK_OK = True
except Exception:  # noqa: BLE001 - headless or no tcl/tk
    _TK_OK = False


class QueueStreamTests(unittest.TestCase):
    """The worker-to-UI stream behaves like a file object."""

    def test_write_enqueues_and_reports_length(self) -> None:
        import queue

        sink: queue.Queue = queue.Queue()
        stream = gui_module._QueueStream(sink, "bench")
        self.assertEqual(stream.write("hello"), 5)
        stream.flush()
        self.assertEqual(sink.get_nowait(), ("bench", "hello"))

    def test_empty_write_is_not_queued(self) -> None:
        import queue

        sink: queue.Queue = queue.Queue()
        gui_module._QueueStream(sink).write("")
        self.assertTrue(sink.empty())


@unittest.skipUnless(_TK_OK, "Tk display not available")
class GUITests(unittest.TestCase):
    """The window builds and its actions work off the UI thread."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_gui_"))
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = gui_module.UltraQuantGUI(self.root, self.dir / "home")
        self.assertTrue(self._wait(), "session did not start")

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - already gone
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def _wait(self, timeout: float = 60.0) -> bool:
        """Pump the event loop until the worker is idle."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if not self.app.busy:
                return True
            time.sleep(0.02)
        return False

    def test_session_and_tabs_are_built(self) -> None:
        self.assertIsNotNone(self.app.session)
        self.assertIsNotNone(self.app.cli)
        tabs = [self.app.notebook.tab(i, "text") for i in range(self.app.notebook.index("end"))]
        self.assertEqual(
            tabs,
            ["Chat", "Learn", "Compute", "Forge", "Library", "Storage",
             "Stash", "Benchmark"],
        )
        self.assertEqual(self.app.status.get(), "ready")

    def test_chat_round_trip_and_trace(self) -> None:
        self.app._submit("the tower height is 324 metres")
        self.assertTrue(self._wait())
        self.app._submit("what is the tower height?")
        self.assertTrue(self._wait())

        transcript = self.app.transcript.get("1.0", "end")
        self.assertIn("324 metres", transcript)
        trace = [line for line in self.app.trace.get("1.0", "end").splitlines() if line.strip()]
        self.assertEqual(len(trace), 6, "one line per thought in the pipeline")
        self.assertTrue(trace[0].startswith("Perceive"))

    def test_input_box_send_clears_and_submits(self) -> None:
        self.app.chat_input.insert("1.0", "calc: 6 * 7")
        self.app._send()
        self.assertTrue(self._wait())
        self.assertEqual(self.app.chat_input.get("1.0", "end").strip(), "")
        self.assertIn("42", self.app.transcript.get("1.0", "end"))

    def test_multiline_input_feeds_glyph_rows(self) -> None:
        """A pasted glyph is five lines; the extra lines must reach the command."""
        from ultraquant.pattern.recognition import PATTERNS

        self.app._submit("\n".join(PATTERNS["plus"]))
        self.assertTrue(self._wait())
        trace = self.app.trace.get("1.0", "end")
        self.assertIn("glyph", trace, "all five rows must reach the pipeline")

    def test_library_and_budget_controls(self) -> None:
        self.app._refresh_library()
        self.app.budget_var.set("64")
        self.app._apply_budget()
        self.assertEqual(self.app.session.cache.stats()["budget_bytes"], 64 * 1024)
        self.assertIn("budget", self.app.residency.get("1.0", "end"))

    def test_stash_table_refreshes(self) -> None:
        self.app._refresh_stash()
        self.assertEqual(len(self.app.stash_view.get_children()), 0)
        self.app._analyze()
        self.assertIn("[stash]", self.app.transcript.get("1.0", "end"))

    def test_learning_mode_asks_and_applies(self) -> None:
        """Find gaps, show a question, answer it, and see the answer land."""
        for _ in range(4):
            self.app._submit("tell me about photonics")
            self.assertTrue(self._wait())

        self.app._survey()
        self.assertTrue(self._wait(timeout=120))
        self.assertTrue(self.app.learner.pending, "survey found no gaps")
        self.assertNotIn("Press 'Find gaps'", self.app.question_var.get())

        question = self.app.learner.next_question()
        while question is not None and question.subject != "photonics":
            self.app._skip_question()
            question = self.app.learner.next_question()
        self.assertIsNotNone(question, "the repeated unknown term was never asked about")

        self.app.answer_input.insert("1.0", "the science of light")
        self.app._answer_question()
        self.assertTrue(self._wait(timeout=120))

        fact = self.app.session.memory.recall_fact("photonics")
        self.assertIsNotNone(fact)
        self.assertIn("light", str(fact["value"]))
        self.assertIn("learned", self.app.learn_log.get("1.0", "end"))

    def test_an_unrecognised_pattern_becomes_a_question_in_the_ui(self) -> None:
        """Chat says it cannot place a shape; the Learn tab asks about it.

        The glyph must be rendered as a shape, not folded into the wrapped
        proportional-font prompt label, or the user cannot see what is being
        asked about.
        """
        self.app._submit("\n".join(["....#", "...#.", "..#..", ".#...", "#...."]))
        self.assertTrue(self._wait(timeout=120))
        transcript = self.app.transcript.get("1.0", "end")
        self.assertNotIn("reads as", transcript, "an unplaceable shape must not be guessed at")

        self.app._survey()
        self.assertTrue(self._wait(timeout=120))
        question = self.app.learner.next_question()
        while question is not None and question.kind != "unknown-pattern":
            self.app._skip_question()
            question = self.app.learner.next_question()
        self.assertIsNotNone(question, "the unplaceable pattern was never asked about")

        prompt = self.app.question_var.get()
        self.assertIn("unknown-pattern", prompt)
        self.assertNotIn("....#", prompt, "the glyph belongs in its own widget")
        shown = [child.cget("text") for child in self.app.option_row.winfo_children()]
        self.assertTrue(any("....#" in str(text) for text in shown),
                        f"the shape was never rendered: {shown}")
        self.assertTrue(any("<category> <label>" in str(text) for text in shown),
                        "the answer format was not explained")

        self.app.answer_input.insert("1.0", "diagonals backslash")
        self.app._answer_question()
        self.assertTrue(self._wait(timeout=120))
        self.assertIn("diagonals", self.app.session.vault.signatures())

    def test_learning_answer_without_a_survey_is_survivable(self) -> None:
        self.app.learner = None
        self.app.answer_input.insert("1.0", "something")
        self.app._answer_question()
        self.assertIn("Find gaps", self.app.status.get())

    def test_storage_tab_inspects_a_backend(self) -> None:
        self.app.storage_uri.set(f"local://{(self.dir / 'inspect').as_posix()}")
        self.app.storage_cache.set("64MB")
        self.app._inspect_storage()
        self.assertTrue(self._wait(timeout=120))
        log = self.app.storage_log.get("1.0", "end")
        self.assertIn("ram-over-local-fs", log)
        self.assertIn("resident_bytes", log)

    def test_status_strip_reports_cpu_and_gpu_without_being_asked(self) -> None:
        """Regression: the strip was blank until something triggered detection."""
        strip = self.app.tiers.get()
        self.assertIn("CPU:", strip)
        self.assertIn("GPU:", strip)
        self.assertNotIn("detecting", strip)

    def test_compute_tab_is_populated_at_startup(self) -> None:
        """It must not sit on 'detecting...' waiting for a Refresh click."""
        self.assertNotIn("detecting", self.app.compute_devices_label.get())
        self.assertIn("CPU:", self.app.compute_devices_label.get())
        self.assertIn("simulator", self.app.quantum_label.get().lower())

    def test_detection_failure_cannot_blank_the_strip(self) -> None:
        """A probe that raises must not leave the user staring at nothing."""
        import ultraquant.gui as gui_module

        original = gui_module.UltraQuantGUI._start_session
        broken = {"called": False}

        def explode(*_a, **_k):
            broken["called"] = True
            raise RuntimeError("probe exploded")

        import ultraquant.forge.trainer as trainer

        saved = trainer.forge_tier_report
        trainer.forge_tier_report = explode
        try:
            self.app._detect_compute()
            self.assertTrue(self._wait(timeout=60))
        finally:
            trainer.forge_tier_report = saved
        self.assertIn("CPU:", self.app.compute_devices_label.get())

    def test_compute_tab_detects_devices(self) -> None:
        self.app._detect_compute()
        self.assertTrue(self._wait(timeout=120))
        self.assertIn("CPU:", self.app.compute_devices_label.get())
        self.assertIn("simulator", self.app.quantum_label.get().lower())

    def test_every_training_device_maps_to_a_trainer_tier(self) -> None:
        from ultraquant.forge.trainer import train_experts
        from ultraquant.gui import TRAIN_DEVICES

        import inspect

        valid = {"auto", "both", "cuda", "cpu", "python"}
        for label, tier in TRAIN_DEVICES:
            with self.subTest(device=label):
                self.app.compute_device.set(label)
                self.assertEqual(self.app._selected_train_tier(), tier)
                self.assertIn(tier, valid)
        self.assertTrue(inspect.signature(train_experts).parameters["tier"])

    def test_every_quantum_tier_runs_a_bell_state(self) -> None:
        """Selecting any tier must produce correct values, falling back if absent."""
        from ultraquant.native.dispatch import QUANTUM_TIERS

        for key, label in QUANTUM_TIERS:
            with self.subTest(tier=key):
                self.app.quantum_tier.set(label)
                self.assertEqual(self.app._selected_quantum_tier(), key)
                self.app._test_quantum()
                self.assertTrue(self._wait(timeout=120))
        log = self.app.compute_log.get("1.0", "end")
        self.assertIn("Bell <Z>", log)
        self.assertNotIn("Traceback", log)

    def test_bad_shots_value_is_refused(self) -> None:
        self.app.quantum_shots.set("many")
        self.app._test_quantum()
        self.assertIn("whole number", self.app.status.get())

    def test_ram_slider_is_bounded_by_available_memory(self) -> None:
        from ultraquant.storage.ram import available_ram

        avail_mb = max(1, (available_ram() or 0) // (1024 * 1024))
        self.app.ram_mb.set(avail_mb * 4)          # try to exceed the machine
        self.app._update_ram_label()
        self.assertLessEqual(
            self.app._ram_avail_mb, self.app._ram_total_mb,
            "available memory cannot exceed total",
        )
        self.assertIn("% of system RAM", self.app.ram_label.get())

    def test_apply_compute_sets_the_budget(self) -> None:
        self.app.ram_mb.set(128)
        self.app._apply_compute()
        self.assertEqual(self.app.session.cache.stats()["budget_bytes"], 128 * 1024 * 1024)
        self.assertIn("RAM tier   : 128 MB", self.app.compute_log.get("1.0", "end"))

    def test_bluequbit_token_is_not_written_to_disk(self) -> None:
        self.app.bluequbit_token.set("test-token-abc")
        self.app._apply_bluequbit_token()
        self.assertTrue(self._wait(timeout=120))
        self.assertEqual(os.environ.get("BLUEQUBIT_API_TOKEN"), "test-token-abc")
        for path in self.dir.rglob("*"):
            if path.is_file():
                try:
                    self.assertNotIn("test-token-abc", path.read_text(errors="ignore"))
                except OSError:
                    pass
        self.app.bluequbit_token.set("")
        self.app._apply_bluequbit_token()
        self.assertTrue(self._wait(timeout=120))
        self.assertIsNone(os.environ.get("BLUEQUBIT_API_TOKEN"))

    def test_backend_picker_composes_every_scheme(self) -> None:
        """Each menu entry must build a URI the registry actually understands."""
        from ultraquant.gui import BACKEND_CHOICES
        from ultraquant.storage import URI_SCHEMES

        for label, scheme in BACKEND_CHOICES:
            with self.subTest(backend=label):
                self.app.storage_backend.set(label)
                self.app._on_backend_change()
                uri = self.app._compose_uri()
                self.assertTrue(uri, f"{label} produced an empty URI")
                if scheme == "custom":
                    continue
                self.assertTrue(
                    uri.startswith(f"{scheme}://"),
                    f"{label} produced {uri!r}, expected the {scheme}:// scheme",
                )
                self.assertIn(scheme, URI_SCHEMES,
                              f"the picker offers {scheme!r} but the registry does not")

    def test_tuning_options_reach_the_uri(self) -> None:
        self.app.storage_backend.set("NVMe-oF namespace")
        self.app._on_backend_change()
        self.app.storage_location.set("X:/uq")
        self.app.storage_qdepth.set("64")
        self.app.storage_readahead.set("32")
        self.app.storage_direct.set(False)
        uri = self.app._compose_uri()
        self.assertIn("queue_depth=64", uri)
        self.assertIn("readahead=32768", uri, "readahead is entered in KB")
        self.assertIn("direct=0", uri)
        self.assertEqual(self.app.storage_uri.get(), uri,
                         "the displayed URI must match what would be opened")

    def test_tuning_is_disabled_where_it_does_not_apply(self) -> None:
        self.app.storage_backend.set("Local filesystem")
        self.app._on_backend_change()
        self.assertIn("disabled", self.app.storage_direct_box.state())
        self.app.storage_backend.set("Pure Storage FlashArray")
        self.app._on_backend_change()
        self.assertNotIn("disabled", self.app.storage_direct_box.state())

    def test_rados_uses_pool_and_namespace(self) -> None:
        self.app.storage_backend.set("Ceph RADOS pool")
        self.app._on_backend_change()
        self.assertEqual(self.app.storage_location_label.get(), "Pool:")
        self.app.storage_location.set("models")
        self.app.storage_extra.set("uq")
        self.assertEqual(self.app._compose_uri(), "rados://models/uq")

    def test_unreachable_location_reports_plainly(self) -> None:
        """A missing drive deserves a sentence, not a stack trace."""
        self.app.storage_backend.set("NVMe-oF namespace")
        self.app._on_backend_change()
        self.app.storage_location.set("Q:/definitely-not-mounted")
        self.app._inspect_storage()
        self.assertTrue(self._wait(timeout=120))
        log = self.app.storage_log.get("1.0", "end")
        self.assertIn("not reachable", log)
        self.assertNotIn("Traceback", log)

    def test_empty_location_is_refused(self) -> None:
        self.app.storage_backend.set("Local filesystem")
        self.app._on_backend_change()
        self.app.storage_location.set("")
        self.app._inspect_storage()
        self.assertIn("location", self.app.status.get())

    def test_array_call_without_an_endpoint_is_refused(self) -> None:
        self.app.array_endpoint.set("")
        self.app._array_connect()
        self.assertIn("endpoint", self.app.status.get())

    def test_forge_runs_and_restores_stdout(self) -> None:
        import sys

        before = sys.stdout
        self.app.forge_vars["categories"].set("4")
        self.app.forge_vars["labels"].set("3")
        self.app.forge_vars["per_class"].set("12")
        self.app.forge_vars["epochs"].set("10")
        self.app.forge_vars["hidden"].set("12")
        self.app._forge()
        self.assertTrue(self._wait(timeout=180))

        self.assertIs(sys.stdout, before, "stdout must be restored after the run")
        log = self.app.forge_log.get("1.0", "end")
        self.assertIn("experts", log)
        self.assertIn("Verified by paging", log)

    def test_busy_guard_rejects_overlapping_work(self) -> None:
        """Only one worker at a time: stdout redirection is process-global."""
        self.app.busy = True
        started = []
        self.app._run_async("second", lambda: started.append(True))
        self.assertEqual(started, [], "a second worker must not start while busy")
        self.app.busy = False

    def test_worker_exception_is_surfaced_not_fatal(self) -> None:
        def boom() -> None:
            raise RuntimeError("forced failure for the test")

        self.app._run_async("failing", boom)
        deadline = time.time() + 20
        while time.time() < deadline and self.app.busy:
            self.app._pump_once()
            time.sleep(0.02)
        self.assertFalse(self.app.busy, "the app must recover from a worker error")
        self.assertIn("forced failure for the test",
                      self.app.transcript.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
