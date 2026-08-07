"""Persistent settings: they survive a restart, and nothing else survives with them.

Every test points ``$ULTRAQUANT_CONFIG`` at a temporary file, so running the
suite can never read or clobber a real user's preferences.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.config import (
    DEFAULTS,
    NEVER_PERSISTED,
    Settings,
    config_path,
)


class _Isolated(unittest.TestCase):
    """A temporary config location, restored afterwards."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_cfg_"))
        self.file = self.dir / "settings.json"
        self._previous = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.file)

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._previous
        shutil.rmtree(self.dir, ignore_errors=True)


class RoundTripTests(_Isolated):
    """The whole point: a setting outlives the process."""

    def test_a_saved_setting_comes_back(self):
        first = Settings.load()
        first.set("budget_bytes", 8 * 1024 * 1024)
        first.set("online", True)
        self.assertTrue(first.save())
        self.assertEqual(Settings.load().get("budget_bytes"), 8 * 1024 * 1024)
        self.assertTrue(Settings.load().get("online"))

    def test_nested_settings_round_trip(self):
        first = Settings.load()
        first.set("lmstudio.panel_models", ["a/b", "c/d"])
        first.set("forge.epochs", 120)
        first.save()
        second = Settings.load()
        self.assertEqual(second.get("lmstudio.panel_models"), ["a/b", "c/d"])
        self.assertEqual(second.get("forge.epochs"), 120)

    def test_untouched_nested_defaults_survive_a_partial_save(self):
        """Setting one key of a group must not wipe its siblings."""
        first = Settings.load()
        first.set("lmstudio.ttl", 60)
        first.save()
        second = Settings.load()
        self.assertEqual(second.get("lmstudio.ttl"), 60)
        self.assertEqual(second.get("lmstudio.quarantine"),
                         DEFAULTS["lmstudio"]["quarantine"])

    def test_first_run_has_no_file_and_no_complaints(self):
        settings = Settings.load()
        self.assertFalse(settings.existed)
        self.assertEqual(settings.problems, [])
        self.assertEqual(settings.get("budget_bytes"),
                         DEFAULTS["budget_bytes"])

    def test_reading_settings_never_writes(self):
        Settings.load()
        self.assertFalse(self.file.exists(),
                         "merely loading must not create a config file")


class DegradationTests(_Isolated):
    """A broken preferences file must never stop the program starting."""

    def test_corrupt_json_falls_back_and_says_so(self):
        self.file.write_text("{ not json at all", encoding="utf-8")
        settings = Settings.load()
        self.assertEqual(settings.get("budget_bytes"),
                         DEFAULTS["budget_bytes"])
        self.assertTrue(any("not valid JSON" in p for p in settings.problems))

    def test_a_json_list_is_not_a_settings_object(self):
        self.file.write_text("[1, 2, 3]", encoding="utf-8")
        settings = Settings.load()
        self.assertTrue(settings.problems)
        self.assertEqual(settings.get("online"), DEFAULTS["online"])

    def test_a_wrongly_typed_value_is_rejected_to_the_default(self):
        """A string where an int belongs would blow up inside a UI callback."""
        self.file.write_text(json.dumps({"budget_bytes": "lots"}),
                             encoding="utf-8")
        settings = Settings.load()
        self.assertEqual(settings.get("budget_bytes"),
                         DEFAULTS["budget_bytes"])
        self.assertTrue(any("budget_bytes" in p for p in settings.problems))

    def test_a_bool_is_not_an_int(self):
        """bool subclasses int, so True would sail into a numeric setting."""
        self.file.write_text(json.dumps({"budget_bytes": True}),
                             encoding="utf-8")
        settings = Settings.load()
        self.assertEqual(settings.get("budget_bytes"),
                         DEFAULTS["budget_bytes"])

    def test_an_int_is_not_a_bool(self):
        self.file.write_text(json.dumps({"online": 1}), encoding="utf-8")
        self.assertEqual(Settings.load().get("online"), DEFAULTS["online"])

    def test_a_nested_bad_value_is_named_by_its_full_path(self):
        self.file.write_text(json.dumps({"forge": {"epochs": "many"}}),
                             encoding="utf-8")
        settings = Settings.load()
        self.assertTrue(any("forge.epochs" in p for p in settings.problems))

    def test_an_unwritable_location_is_survivable(self):
        settings = Settings.load()
        settings.path = self.dir / "nope" / "\0bad" / "settings.json"
        self.assertFalse(settings.save())
        self.assertTrue(settings.problems)

    def test_a_failed_save_leaves_no_scratch_file(self):
        settings = Settings.load()
        settings.set("online", True)
        settings.save()
        leftovers = [p.name for p in self.dir.iterdir()
                     if p.name.startswith(".uqcfg-")]
        self.assertEqual(leftovers, [])

    def test_a_save_does_not_destroy_the_previous_file_on_failure(self):
        """Atomic write: an interrupted save must not leave a corrupt file."""
        first = Settings.load()
        first.set("budget_bytes", 4096)
        first.save()
        before = self.file.read_text(encoding="utf-8")
        second = Settings.load()
        second.values["forge"] = {1: object()}      # not JSON-serialisable
        second.save()
        self.assertEqual(self.file.read_text(encoding="utf-8"), before)


class SecretTests(_Isolated):
    """Credentials never reach a plaintext file that may sit in a checkout."""

    def test_setting_a_credential_is_refused_loudly(self):
        settings = Settings.load()
        for key in ("bluequbit_token", "array_secret", "lmstudio.password"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    settings.set(key, "hunter2")

    def test_a_credential_is_stripped_on_save(self):
        settings = Settings.load()
        settings.values["bluequbit_token"] = "sk-leaked"   # bypass set()
        settings.save()
        self.assertNotIn("sk-leaked", self.file.read_text(encoding="utf-8"))

    def test_a_hand_edited_credential_is_ignored_on_load(self):
        self.file.write_text(json.dumps({"bluequbit_token": "sk-leaked"}),
                             encoding="utf-8")
        settings = Settings.load()
        self.assertIsNone(settings.get("bluequbit_token"))
        self.assertTrue(any("credentials are never read" in p
                            for p in settings.problems))

    def test_a_nested_credential_is_stripped_too(self):
        """Stripping only the top level would leak anything one level down."""
        settings = Settings.load()
        settings.values["lmstudio"]["api_key"] = "sk-leaked"
        settings.save()
        self.assertNotIn("sk-leaked", self.file.read_text(encoding="utf-8"))

    def test_describe_redacts_a_nested_credential(self):
        settings = Settings.load()
        settings.values["lmstudio"]["token"] = "sk-leaked"
        described = settings.describe()
        self.assertNotIn("sk-leaked", described)
        self.assertIn("redacted", described)

    def test_describe_never_prints_a_credential(self):
        settings = Settings.load()
        settings.values["api_key"] = "sk-leaked"
        self.assertNotIn("sk-leaked", settings.describe())

    def test_every_named_secret_is_actually_refused(self):
        settings = Settings.load()
        for key in NEVER_PERSISTED:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    settings.set(key, "x")


class ForwardCompatibilityTests(_Isolated):
    """Old code must not silently delete a newer build's settings."""

    def test_unknown_keys_survive_a_load_and_save(self):
        self.file.write_text(json.dumps({
            "budget_bytes": 2048,
            "a_setting_from_the_future": {"nested": [1, 2]},
        }), encoding="utf-8")
        settings = Settings.load()
        settings.set("online", True)
        settings.save()
        written = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(written["a_setting_from_the_future"],
                         {"nested": [1, 2]})

    def test_a_version_is_always_written(self):
        Settings.load().save()
        self.assertIn("version",
                      json.loads(self.file.read_text(encoding="utf-8")))


class LocationTests(unittest.TestCase):
    """Where the file goes, and why."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_loc_"))
        self._env = os.environ.get("ULTRAQUANT_CONFIG")
        self._cwd = Path.cwd()

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        if self._env is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._env
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_environment_override_wins(self):
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "explicit.json")
        self.assertEqual(config_path(), self.dir / "explicit.json")

    def test_a_local_file_makes_the_install_portable(self):
        os.environ.pop("ULTRAQUANT_CONFIG", None)
        (self.dir / "ultraquant.json").write_text("{}", encoding="utf-8")
        os.chdir(self.dir)
        self.assertEqual(config_path().name, "ultraquant.json")
        self.assertEqual(config_path().parent.resolve(), self.dir.resolve())

    def test_without_a_local_file_it_uses_the_user_directory(self):
        """Otherwise an ordinary run would scatter config files around."""
        os.environ.pop("ULTRAQUANT_CONFIG", None)
        os.chdir(self.dir)
        self.assertNotEqual(config_path().parent.resolve(), self.dir.resolve())


class UnitTests(_Isolated):
    """The budget is stored in bytes because the surfaces disagree on units."""

    def test_budget_is_documented_in_bytes(self):
        """The GUI field is KB and the TUI command is MB; the file is bytes.

        One config key in either surface's own unit would apply a 1024x error
        when the other read it back.
        """
        self.assertEqual(DEFAULTS["budget_bytes"], 1024 * 1024)
        self.assertNotIn("budget_mb", DEFAULTS)
        self.assertNotIn("budget_kb", DEFAULTS)


class SurfacePersistenceTests(_Isolated):
    """The user-visible promise: a setting changed on a surface comes back."""

    def test_the_tui_remembers_a_budget_across_a_restart(self):
        import io

        from ultraquant.tui import UltraQuantTUI

        home = self.dir / "home"
        first = UltraQuantTUI(home=home, stream=io.StringIO(), color=False)
        first.handle(":compute ram 7")
        second = UltraQuantTUI(home=home, stream=io.StringIO(), color=False)
        second.ensure_session()
        self.assertEqual(second.session.cache.stats()["budget_bytes"],
                         7 * 1024 * 1024)

    def test_the_tui_remembers_the_network_gate(self):
        import io

        from ultraquant.tui import UltraQuantTUI

        home = self.dir / "home2"
        first = UltraQuantTUI(home=home, stream=io.StringIO(), color=False)
        first.ensure_session()
        first.session.web.set_online(True)
        first._save_settings()
        second = UltraQuantTUI(home=home, stream=io.StringIO(), color=False)
        second.ensure_session()
        self.assertTrue(second.session.web.online)

    def test_an_explicit_home_beats_the_saved_one(self):
        """A path given on the command line must never be overridden."""
        import io

        from ultraquant.tui import UltraQuantTUI

        settings = Settings.load()
        settings.set("home", str(self.dir / "saved"))
        settings.save()
        explicit = self.dir / "explicit"
        tui = UltraQuantTUI(home=explicit, stream=io.StringIO(), color=False)
        self.assertEqual(tui.home, explicit)

    def test_the_saved_home_is_used_when_none_is_given(self):
        import io

        from ultraquant.tui import UltraQuantTUI

        settings = Settings.load()
        settings.set("home", str(self.dir / "saved"))
        settings.save()
        tui = UltraQuantTUI(stream=io.StringIO(), color=False)
        self.assertEqual(tui.home, self.dir / "saved")


class SchemaTypeTests(_Isolated):
    """Widget values must be stored in the type the schema declares.

    The GUI holds several numbers in StringVars. Writing "99" where the schema
    says int was rejected on the next load and the setting silently reverted -
    the failure this pair of helpers exists to prevent.
    """

    def test_a_numeric_string_is_stored_as_a_number(self):
        from ultraquant.gui import _stored_value

        self.assertEqual(_stored_value("forge.epochs", "99"), 99)
        self.assertEqual(_stored_value("quantum_shots", "512"), 512)

    def test_a_text_setting_stays_text(self):
        from ultraquant.gui import _stored_value

        self.assertEqual(_stored_value("forge.tier", "cpu"), "cpu")

    def test_a_stored_number_round_trips_through_the_schema(self):
        from ultraquant.gui import _stored_value

        settings = Settings.load()
        settings.set("forge.epochs", _stored_value("forge.epochs", "99"))
        settings.save()
        reloaded = Settings.load()
        self.assertEqual(reloaded.get("forge.epochs"), 99)
        self.assertEqual(reloaded.problems, [],
                         "a well-typed value must not be rejected on load")


if __name__ == "__main__":
    unittest.main()
