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
             "Stash", "Panel", "Settings", "Benchmark"],
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


class PanelTabTests(unittest.TestCase):
    """The LLMLS tab, driven with a stub catalogue.

    No LM Studio is contacted. The tab's job is to make the independence
    accounting *visible before a question is asked* - selecting four Llama
    derivatives must read "1 voice" up front, not come back looking
    mysteriously uncorroborated afterwards - and that is what these check.
    """

    def setUp(self) -> None:
        from ultraquant.interpreter.llmls import ModelCard, independent_groups

        self.dir = Path(tempfile.mkdtemp(prefix="uq_panel_"))
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = gui_module.UltraQuantGUI(self.root, self.dir / "home")
        deadline = time.time() + 60
        while time.time() < deadline and self.app.busy:
            self.root.update()
            time.sleep(0.02)

        self.cards = [
            ModelCard("qwen/qwen3-coder-30b", "qwen3moe", "qwen", loaded=True),
            ModelCard("qwen/qwen3-coder-next", "qwen3next", "qwen"),
            ModelCard("openai/gpt-oss-20b", "gpt-oss", "openai"),
            ModelCard("google/gemma-4-31b", "gemma4", "google", kind="vlm"),
        ]
        self.app._fill_panel_tree(independent_groups(self.cards))
        self.root.update()

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - already gone
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def _select(self, *model_ids: str) -> None:
        wanted = []
        for parent in self.app.panel_tree.get_children():
            for child in self.app.panel_tree.get_children(parent):
                if self.app.panel_tree.item(child, "text") in model_ids:
                    wanted.append(child)
        self.app.panel_tree.selection_set(wanted)
        self.app._panel_selection_changed()
        self.root.update()

    def test_the_tree_groups_models_under_their_voice(self):
        voices = self.app.panel_tree.get_children()
        self.assertEqual(len(voices), 3, "four models, three lineages")

    def test_a_shared_lineage_is_labelled_as_one_source(self):
        labels = [self.app.panel_tree.item(v, "text")
                  for v in self.app.panel_tree.get_children()]
        self.assertTrue(any("ONE source" in label for label in labels),
                        f"the Qwen pair should be marked: {labels}")

    def test_selecting_two_siblings_reports_one_voice(self):
        """The number that matters, shown before the question is asked."""
        self._select("qwen/qwen3-coder-30b", "qwen/qwen3-coder-next")
        text = self.app.panel_selection.get()
        self.assertIn("2 model(s) selected = 1 independent voice", text)
        self.assertIn("cannot corroborate itself", text)

    def test_selecting_across_lineages_reports_several_voices(self):
        self._select("qwen/qwen3-coder-30b", "openai/gpt-oss-20b",
                     "google/gemma-4-31b")
        self.assertIn("3 model(s) selected = 3 independent voice(s)",
                      self.app.panel_selection.get())

    def test_voice_header_rows_are_never_treated_as_models(self):
        """Selecting a header must not send a fake model id to LM Studio."""
        self.app.panel_tree.selection_set(
            list(self.app.panel_tree.get_children()))
        self.assertEqual(self.app._panel_models(), [])

    def test_loaded_state_is_shown(self):
        states = []
        for parent in self.app.panel_tree.get_children():
            for child in self.app.panel_tree.get_children(parent):
                states.append((self.app.panel_tree.item(child, "text"),
                               self.app.panel_tree.set(child, "state")))
        self.assertIn(("qwen/qwen3-coder-30b", "loaded"), states)

    def test_asking_with_nothing_selected_is_refused(self):
        self.app.panel_question.set("what is the capital of Australia?")
        self.app.panel_tree.selection_set([])
        self.app._panel_ask()
        self.root.update()
        self.assertFalse(self.app.busy, "no worker should have started")

    def test_asking_with_no_question_is_refused(self):
        self._select("qwen/qwen3-coder-30b")
        self.app.panel_question.set("   ")
        self.app._panel_ask()
        self.root.update()
        self.assertFalse(self.app.busy)

    def test_settings_tab_edits_persist_and_reach_the_session(self):
        """Type into the tab, apply, and the change is on disk and live."""
        import os

        from ultraquant.config import Settings

        previous = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "tab.json")
        try:
            self.app.settings = Settings.load()
            self.app.settings_editors["budget_bytes"][0].set("2097152")
            self.app.settings_editors["online"][0].set(True)
            self.app.settings_editors["forge.epochs"][0].set("77")
            self.app._settings_apply()
            self.root.update()
            self.assertIn("saved", self.app.settings_note.get())
            reloaded = Settings.load()
            self.assertEqual(reloaded.get("budget_bytes"), 2097152)
            self.assertEqual(reloaded.get("forge.epochs"), 77)
            self.assertTrue(reloaded.get("online"))
            self.assertEqual(
                self.app.session.cache.stats()["budget_bytes"], 2097152,
                "applying settings must reach the running session")
        finally:
            if previous is None:
                os.environ.pop("ULTRAQUANT_CONFIG", None)
            else:
                os.environ["ULTRAQUANT_CONFIG"] = previous

    def test_every_schema_key_is_editable(self):
        """The tab is schema-driven, so a new setting cannot go missing.

        A hand-written second list of widgets is exactly how one surface ends
        up able to change a setting the other cannot see.
        """
        from ultraquant.config import DEFAULTS, NEVER_PERSISTED

        expected = set()
        for key, default in DEFAULTS.items():
            if key == "version" or key in NEVER_PERSISTED:
                continue
            if isinstance(default, dict):
                expected.update(f"{key}.{sub}" for sub in default
                                if sub not in NEVER_PERSISTED)
            else:
                expected.add(key)
        self.assertEqual(set(self.app.settings_editors), expected)

    def test_an_unreadable_field_blocks_the_whole_save(self):
        """A partial save would leave the file disagreeing with the screen."""
        import os

        from ultraquant.config import Settings

        previous = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "bad.json")
        try:
            self.app.settings = Settings.load()
            self.app.settings_editors["budget_bytes"][0].set("not a number")
            self.app.settings_editors["forge.epochs"][0].set("55")
            self.app._settings_apply()
            self.root.update()
            self.assertIn("not saved", self.app.settings_note.get())
            self.assertIn("budget_bytes", self.app.settings_note.get())
            self.assertFalse((self.dir / "bad.json").exists(),
                             "nothing may be written when a field is bad")
        finally:
            if previous is None:
                os.environ.pop("ULTRAQUANT_CONFIG", None)
            else:
                os.environ["ULTRAQUANT_CONFIG"] = previous

    def test_a_list_setting_is_edited_as_comma_separated_text(self):
        var, _default = self.app.settings_editors["lmstudio.panel_models"]
        var.set("qwen/a, openai/b")
        self.assertEqual(
            gui_module._editor_parse(var.get(), []), ["qwen/a", "openai/b"])

    def test_no_credential_is_editable_in_the_tab(self):
        """They are not stored, so offering a field for them would mislead."""
        from ultraquant.config import NEVER_PERSISTED

        for key in self.app.settings_editors:
            leaf = key.split(".")[-1]
            self.assertNotIn(leaf, NEVER_PERSISTED)

    def test_reset_does_not_write_until_applied(self):
        """Reset is easy to hit by accident; it must be undoable."""
        import os

        from ultraquant.config import DEFAULTS, Settings

        previous = os.environ.get("ULTRAQUANT_CONFIG")
        target = self.dir / "reset.json"
        os.environ["ULTRAQUANT_CONFIG"] = str(target)
        try:
            self.app.settings = Settings.load()
            self.app.settings.set("forge.epochs", 500)
            self.app.settings.save()
            self.app._settings_reset()
            self.root.update()
            self.assertEqual(Settings.load().get("forge.epochs"), 500,
                             "reset must not touch the file on its own")
            self.assertEqual(
                self.app.settings_editors["forge.epochs"][0].get(),
                str(DEFAULTS["forge"]["epochs"]))
        finally:
            if previous is None:
                os.environ.pop("ULTRAQUANT_CONFIG", None)
            else:
                os.environ["ULTRAQUANT_CONFIG"] = previous

    def test_a_corroborated_answer_is_filled_in_but_never_applied(self):
        """The whole point of the wiring: review, then apply. Not apply.

        A panel that submitted its own answers would be an oracle, which is
        the arrangement this module refuses. The answer box gets the agreed
        position and the user still has to press Answer.
        """
        self.app.answer_input.delete("1.0", "end")
        applied = []
        self.app._answer_question = lambda *a, **k: applied.append(1)
        self.app.events.put(("learn_suggest", "canberra"))
        self.app._pump_once()
        self.root.update()
        self.assertEqual(self.app.answer_input.get("1.0", "end").strip(),
                         "canberra")
        self.assertEqual(applied, [], "nothing may be submitted automatically")

    def test_a_suggestion_replaces_rather_than_appends(self):
        self.app.answer_input.delete("1.0", "end")
        self.app.answer_input.insert("1.0", "stale text")
        self.app.events.put(("learn_suggest", "au"))
        self.app._pump_once()
        self.root.update()
        self.assertEqual(self.app.answer_input.get("1.0", "end").strip(), "au")

    def test_asking_the_panel_before_surveying_is_refused(self):
        self.app.learner = None
        self.app._learn_ask_panel()
        self.root.update()
        self.assertFalse(self.app.busy)

    def test_asking_the_panel_with_no_models_selected_is_refused(self):
        class _Question:
            expects = "text"
            prompt = "what is the capital of Australia?"

        class _Learner:
            def next_question(self):
                return _Question()

        self.app.learner = _Learner()
        self.app.panel_tree.selection_set([])
        self.app._learn_ask_panel()
        self.root.update()
        self.assertFalse(self.app.busy, "no worker without a panel selection")

    def test_a_glyph_question_is_refused_by_the_text_panel(self):
        class _Question:
            expects = "glyph"
            prompt = "show me a square"

        class _Learner:
            def next_question(self):
                return _Question()

        self.app.learner = _Learner()
        self._select("qwen/qwen3-coder-30b", "openai/gpt-oss-20b")
        self.app._learn_ask_panel()
        self.root.update()
        self.assertFalse(self.app.busy, "pixels cannot come from a text panel")

    def test_the_post_load_refresh_is_not_swallowed_by_the_busy_flag(self):
        """A worker's own follow-up event fires before its 'done' event.

        The reload was queued directly, so it ran while busy was still set and
        _run_async refused it with a spurious "Busy" popup - leaving the tree
        stale after a load. It is deferred one tick instead.
        """
        calls = []
        self.app._panel_refresh = lambda: calls.append(1)
        self.app.busy = True
        self.app.events.put(("panel_reload", None))
        self.app._pump_once()
        self.assertEqual(calls, [], "must not run while the worker is busy")
        self.app.busy = False
        deadline = time.time() + 5
        while time.time() < deadline and not calls:
            self.root.update()
            time.sleep(0.02)
        self.assertEqual(calls, [1], "refresh should land on a later tick")

    def test_an_unreachable_server_reports_rather_than_crashes(self):
        import os

        from ultraquant.interpreter import lmstudio

        previous = os.environ.get("LMSTUDIO_BASE_URL")
        os.environ["LMSTUDIO_BASE_URL"] = "http://127.0.0.1:9/v1"
        try:
            self.app._panel_refresh()
            deadline = time.time() + 30
            while time.time() < deadline and self.app.busy:
                self.root.update()
                time.sleep(0.02)
            self.root.update()
            self.assertIn("unavailable", self.app.panel_status.get())
        finally:
            if previous is None:
                os.environ.pop("LMSTUDIO_BASE_URL", None)
            else:
                os.environ["LMSTUDIO_BASE_URL"] = previous


@unittest.skipUnless(_TK_OK, "Tk display not available")
class DistillTests(unittest.TestCase):
    """The Panel tab's distillation surface: the CLI contract, clickable.

    No LM Studio is contacted: the trainer is stubbed, because what the
    GUI owes is the wiring - session in, report to the log, router saved
    only when something was taught, ledger label refreshed.
    """

    def setUp(self) -> None:
        import queue as queue_module

        self.dir = Path(tempfile.mkdtemp(prefix="uq_distill_"))
        self._old_config = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "settings.json")
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = gui_module.UltraQuantGUI(self.root, self.dir / "home")
        deadline = time.time() + 60
        while time.time() < deadline and self.app.busy:
            self.root.update()
            time.sleep(0.02)
        self.queue_module = queue_module

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._old_config
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - already gone
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def _drain(self, seconds: float = 5.0) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline and self.app.busy:
            self.root.update()
            time.sleep(0.02)
        self.root.update()

    def test_the_ledger_label_reads_the_cumulative_state(self) -> None:
        import json

        vault = self.dir / "home" / "vault"
        vault.mkdir(parents=True, exist_ok=True)
        (vault / "distilled.json").write_text(json.dumps({
            "teachers": ["a-7b", "b-9b"],
            "phrasings": {"lines": ["x"] * 10, "arrows": ["y"] * 8},
        }), encoding="utf-8")
        self.app._distill_refresh()
        label = self.app.distill_status.get()
        self.assertIn("2 voice(s) taught", label)
        self.assertIn("8-10/12", label)

    def test_a_taught_run_reaches_the_log_and_saves_the_router(self) -> None:
        from ultraquant.forge import train_from_llm as trainer

        saved = []
        report = trainer.TrainingReport()
        report.model = "stub-7b"
        report.trained = {"lines": 2}
        report.before, report.after = 0.5, 0.75
        report.decoys_before = report.decoys_after = 1.0

        real_train = trainer.train_next_voice
        real_save = self.app.session.router.save
        trainer.train_next_voice = lambda session, home, **kw: report
        self.app.session.router.save = lambda: saved.append(True)
        try:
            self.app._distill_next()
            self._drain()
        finally:
            trainer.train_next_voice = real_train
            self.app.session.router.save = real_save

        log = self.app.panel_log.get("1.0", "end")
        self.assertIn("stub-7b", log)
        self.assertIn("router saved", log)
        self.assertEqual(saved, [True])

    def test_an_empty_run_saves_nothing(self) -> None:
        """The CLI's persistence rule, kept: no teaching, no save."""
        from ultraquant.forge import train_from_llm as trainer

        saved = []
        report = trainer.TrainingReport()
        report.skipped["(all)"] = "every independent voice has already taught"

        real_train = trainer.train_next_voice
        real_save = self.app.session.router.save
        trainer.train_next_voice = lambda session, home, **kw: report
        self.app.session.router.save = lambda: saved.append(True)
        try:
            self.app._distill_next()
            self._drain()
        finally:
            trainer.train_next_voice = real_train
            self.app.session.router.save = real_save

        self.assertEqual(saved, [])
        self.assertIn("already taught",
                      self.app.panel_log.get("1.0", "end"))

    def test_lm_studio_down_is_a_message_not_a_traceback(self) -> None:
        from ultraquant.forge import train_from_llm as trainer
        from ultraquant.interpreter.lmstudio import LMStudioUnavailable

        def down(session, home, **kw):
            raise LMStudioUnavailable("nobody home")

        real_train = trainer.train_next_voice
        trainer.train_next_voice = down
        try:
            self.app._distill_next()
            self._drain()
        finally:
            trainer.train_next_voice = real_train

        log = self.app.panel_log.get("1.0", "end")
        self.assertIn("LM Studio unavailable: nobody home", log)
        self.assertIsNone(self.app.last_error)


class SkipGroupingTests(unittest.TestCase):
    """Identical skip reasons collapse; distinct ones stay itemised."""

    def test_a_wall_of_identical_skips_becomes_one_line(self) -> None:
        from ultraquant.forge.train_from_llm import TrainingReport

        report = TrainingReport()
        report.model = "stub"
        report.trained = {"lines": 1}
        report.before = report.after = 0.5
        report.decoys_before = report.decoys_after = 1.0
        for index in range(24):
            report.skipped[f"family_{index:03d}"] = (
                "synthetic family, no real topic")
        report.skipped["world"] = "only 1 new usable phrasing(s)"
        text = report.as_text()
        self.assertIn("family_000 .. family_023 (24) skipped:", text)
        self.assertNotIn("family_007", text)
        self.assertIn("world", text)


if __name__ == "__main__":
    unittest.main()
