"""Persistent settings, shared by the GUI, the TUI and the chat CLI.

Every surface previously started from hard-coded defaults, so a RAM budget, a
storage URI or a chosen panel had to be set again on every launch. This is the
one file all of them read on load and write on change.

**Where it lives**, in order:

1. ``$ULTRAQUANT_CONFIG`` — an explicit path, which is also what tests use so
   they never touch a real user's settings.
2. ``./ultraquant.json`` — *portable mode*. A copy of this project on a USB
   stick, or a checkout with its own config beside it, keeps its settings with
   it. Only used when the file already exists, so an ordinary run in some
   unrelated directory does not scatter config files around.
3. The per-user location: ``%APPDATA%\\UltraQuant\\settings.json`` on Windows,
   ``$XDG_CONFIG_HOME/ultraquant/settings.json`` (or ``~/.config/...``)
   elsewhere. This is where a normal install saves.

Note that ``home`` is itself a setting, so the config cannot live inside it —
finding the file would require knowing the value the file contains.

**Secrets are deliberately not stored.** The GUI holds a BlueQubit API token and
an array password. Writing those into a plaintext JSON file, in the user's
config directory or worse in a portable checkout that might be committed, would
turn a convenience feature into a credential leak. :data:`NEVER_PERSISTED` names
them, :func:`Settings.set` refuses them, and they stay where they already were —
in the environment, re-entered per session.

**A broken config must never stop the program starting.** Corrupt JSON, a
directory where a file should be, a value of the wrong type: each degrades to
the default and records a note in :attr:`Settings.problems` for the UI to show.
Refusing to launch because a preferences file is malformed would be a far worse
failure than losing the preferences.

**Unknown keys survive.** A settings file written by a newer build is loaded,
edited and saved without dropping the keys this build does not recognise. Older
code silently deleting newer settings is a nasty, hard-to-see bug.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "Settings",
    "DEFAULTS",
    "NEVER_PERSISTED",
    "config_path",
    "load",
]

#: Bumped only when a migration is needed; :meth:`Settings.load` records what it
#: read so a future version can tell an old file from a new one.
SCHEMA_VERSION = 1

#: Keys that must never reach the file, whatever a caller asks for. These are
#: credentials; the config file is plaintext and may sit in a portable checkout.
NEVER_PERSISTED = frozenset({
    "bluequbit_token", "array_secret", "array_user", "api_key", "token",
    "password", "secret",
})

#: Every persisted setting and its default. A key absent here is still loaded
#: and saved (see the module docstring on forward compatibility) but has no
#: default and no type coercion.
DEFAULTS: dict[str, Any] = {
    "version": SCHEMA_VERSION,
    # --- session -------------------------------------------------------
    "home": "",                       # "" means "use the caller's choice"
    # Bytes, not KB or MB. The GUI's field is in kilobytes and the TUI's
    # `ram` command is in megabytes - one config key in either unit would
    # silently apply a 1024x error when the other surface read it back. Bytes
    # is also what ShardCache.set_budget actually takes, so each surface
    # converts once at its own edge and the stored value is unambiguous.
    "budget_bytes": 1024 * 1024,
    "online": False,
    # --- compute -------------------------------------------------------
    "compute_device": "auto",
    "compute_threads": 0,             # 0 = decide from the machine
    "quantum_tier": "",
    "quantum_shots": 0,
    # --- storage -------------------------------------------------------
    "storage_uri": "",
    "library_root": "",
    "forge_root": "",
    # --- forge defaults ------------------------------------------------
    "forge": {
        "source": "synthetic",
        "categories": 64,
        "labels": 4,
        "per_class": 40,
        "epochs": 40,
        "hidden": 32,
        "tier": "auto",
    },
    # --- LM Studio / LLMLS ---------------------------------------------
    "lmstudio": {
        "base_url": "",               # "" = the client's own default
        "panel_models": [],
        "ttl": 900,
        "gpu": "",                    # "" = let LM Studio decide
        "quarantine": True,
    },
}

#: Types that survive a JSON round trip and are worth coercing on read.
_COERCE = {int: int, float: float, bool: bool, str: str, list: list, dict: dict}


def _user_config_dir() -> Path:
    """The per-user config directory, following the platform convention."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "UltraQuant"
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "ultraquant"


def config_path() -> Path:
    """Where settings are read from and written to.

    Returns:
        The first of: ``$ULTRAQUANT_CONFIG``; ``./ultraquant.json`` when it
        already exists (portable mode); otherwise the per-user location. The
        directory is not created here — :meth:`Settings.save` does that, so
        merely reading settings never writes to disk.
    """
    override = os.environ.get("ULTRAQUANT_CONFIG")
    if override:
        return Path(override)
    portable = Path.cwd() / "ultraquant.json"
    if portable.exists():
        return portable
    return _user_config_dir() / "settings.json"


def _merge(defaults: dict, loaded: dict, problems: list, prefix: str = "") -> dict:
    """Overlay ``loaded`` onto ``defaults``, keeping unknown keys.

    A value whose type does not match the default is rejected to the default
    and noted, because a string where an int belongs would otherwise surface as
    a TypeError deep inside a UI callback rather than as a bad setting.
    """
    merged = copy.deepcopy(defaults)
    for key, value in loaded.items():
        where = f"{prefix}{key}"
        if key not in defaults:
            merged[key] = value          # forward compatibility: keep it
            continue
        default = defaults[key]
        if isinstance(default, dict) and isinstance(value, dict):
            merged[key] = _merge(default, value, problems, f"{where}.")
            continue
        # bool is a subclass of int; check it first or True passes as an int.
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int) and not isinstance(default, bool):
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, type(default))
        if ok:
            merged[key] = value
        else:
            problems.append(
                f"{where}: expected {type(default).__name__}, got "
                f"{type(value).__name__}; using {default!r}"
            )
    return merged


def _strip_secrets(node: Any) -> Any:
    """Drop credential keys at every depth, not only the top level."""
    if not isinstance(node, dict):
        return node
    return {k: _strip_secrets(v) for k, v in node.items()
            if k not in NEVER_PERSISTED}


class Settings:
    """Loaded settings, with the problems found while loading.

    Attributes:
        values: The merged settings.
        path: Where they came from, and where :meth:`save` writes.
        problems: Human-readable notes about anything that had to be defaulted.
            Empty on a clean load. Surfaced by the UIs rather than logged and
            forgotten, since a silently ignored setting looks like a bug in the
            program.
        existed: Whether a file was actually read. False on first run.
    """

    __slots__ = ("values", "path", "problems", "existed")

    def __init__(self, values: dict | None = None, path: Path | None = None,
                 problems: list | None = None, existed: bool = False) -> None:
        self.values = values if values is not None else copy.deepcopy(DEFAULTS)
        self.path = Path(path) if path is not None else config_path()
        self.problems: list[str] = list(problems or [])
        self.existed = existed

    # ---- reading --------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Settings":
        """Read settings, degrading to defaults rather than failing.

        Args:
            path: Override the location. Defaults to :func:`config_path`.

        Returns:
            A :class:`Settings`. Never raises — a missing file is the normal
            first-run case, and a corrupt one is a problem to report, not a
            reason to refuse to start.
        """
        where = Path(path) if path is not None else config_path()
        problems: list[str] = []
        try:
            raw = where.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(copy.deepcopy(DEFAULTS), where, problems, existed=False)
        except OSError as exc:
            problems.append(f"could not read {where}: {exc}; using defaults")
            return cls(copy.deepcopy(DEFAULTS), where, problems, existed=False)
        try:
            loaded = json.loads(raw)
        except ValueError as exc:
            problems.append(f"{where} is not valid JSON ({exc}); using defaults")
            return cls(copy.deepcopy(DEFAULTS), where, problems, existed=True)
        if not isinstance(loaded, dict):
            problems.append(f"{where} does not contain a settings object; "
                            "using defaults")
            return cls(copy.deepcopy(DEFAULTS), where, problems, existed=True)
        values = _merge(DEFAULTS, loaded, problems)
        # A credential that reached the file previously is dropped on read as
        # well as on write, so an old or hand-edited file cannot reintroduce it.
        for key in list(values):
            if key in NEVER_PERSISTED:
                problems.append(f"{key} found in {where.name} and ignored - "
                                "credentials are never read from the config")
                values.pop(key)
        return cls(values, where, problems, existed=True)

    def get(self, key: str, default: Any = None) -> Any:
        """One setting, dotted for nested keys (``"lmstudio.ttl"``)."""
        node: Any = self.values
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ---- writing --------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Change one setting in memory. Dotted keys create nested dicts.

        Raises:
            ValueError: If the key is a credential. Refused loudly rather than
                dropped quietly, so a caller cannot believe a token was saved.
        """
        parts = key.split(".")
        if any(part in NEVER_PERSISTED for part in parts):
            raise ValueError(
                f"refusing to persist {key!r}: the config file is plaintext and "
                "may live in a portable checkout. Keep credentials in the "
                "environment."
            )
        node = self.values
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def update(self, mapping: dict) -> None:
        """Apply several dotted settings at once."""
        for key, value in mapping.items():
            self.set(key, value)

    def save(self) -> bool:
        """Write the settings out atomically.

        Written to a temporary file in the same directory and then moved into
        place, so an interrupted save leaves the previous settings intact rather
        than a half-written file that fails to parse on next launch.

        Returns:
            True on success. False when the location is not writable — a
            read-only install or a locked directory is a reason to keep running
            without persistence, not to fail.
        """
        payload = _strip_secrets(self.values)
        payload["version"] = SCHEMA_VERSION
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False,
                dir=str(self.path.parent), prefix=".uqcfg-", suffix=".tmp")
            try:
                with handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(handle.name, self.path)
            except BaseException:
                # Never leave the scratch file behind on a failed save.
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
                raise
        except (OSError, ValueError, TypeError) as exc:
            # OSError: unwritable location. ValueError: a path the
            # platform rejects outright (an embedded NUL makes
            # mkdir raise ValueError, not OSError, on Windows).
            # TypeError: a value that is not JSON-serialisable,
            # which json.dump raises mid-write - the previous file
            # is still intact because the write went to a temp file.
            self.problems.append(f"could not save {self.path}: {exc}")
            return False
        return True

    # ---- reporting ------------------------------------------------------

    def describe(self) -> str:
        """A printable summary, safe to show anywhere. Never includes secrets.

        Redacted here as well as on read and write. Those two paths keep
        credentials out of the *file*, but a value set directly on
        :attr:`values` — or carried in from a build that named a secret
        differently — would still reach a screen, a log or a bug report through
        this method. Three independent filters is the right number for the one
        thing that must not leak.
        """
        lines = [f"settings: {self.path}"
                 + ("" if self.existed else "  (not created yet)")]
        for key in sorted(self.values):
            if key == "version":
                continue
            if key in NEVER_PERSISTED:
                lines.append(f"  {key:<16} <redacted, never stored>")
                continue
            value = self.values[key]
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub in sorted(value):
                    shown = ("<redacted, never stored>"
                             if sub in NEVER_PERSISTED else repr(value[sub]))
                    lines.append(f"    {sub:<14} {shown}")
            else:
                lines.append(f"  {key:<16} {value!r}")
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


def load(path: str | os.PathLike | None = None) -> Settings:
    """Shorthand for :meth:`Settings.load`."""
    return Settings.load(path)
