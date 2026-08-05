"""ArTchive — T-numbered, tamper-evident temporal snapshots.

The Ar(T)chive stores JSON payload snapshots under sequential version ids
(``"T-0001"``, ``"T-0002"``, ...).  Every snapshot's sha256 is recorded in a
manifest so that any on-disk tampering is detected on restore.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


class IntegrityError(RuntimeError):
    """Raised when a snapshot's bytes no longer match its manifest sha256."""


class ArTchive:
    """Tamper-evident archive of JSON-serializable payload snapshots.

    Layout on disk::

        root/
          manifest.json          {"versions": [{id, label, timestamp, sha256, path}, ...]}
          snapshots/
            T-0001.json
            T-0002.json
            ...
    """

    def __init__(self, root: str | os.PathLike) -> None:
        """Create (or reopen) an archive rooted at ``root``.

        Creates ``root/`` and ``root/snapshots/`` if missing, and creates an
        empty manifest (``{"versions": []}``) when none exists yet.

        Args:
            root: Directory that holds the manifest and the snapshots folder.
        """
        self.root: Path = Path(root)
        self.snapshots_dir: Path = self.root / "snapshots"
        self.manifest_path: Path = self.root / "manifest.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self._manifest: dict = json.load(f)
            if "versions" not in self._manifest:
                self._manifest["versions"] = []
        else:
            self._manifest = {"versions": []}
            self._write_manifest()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _write_manifest(self) -> None:
        """Persist the in-memory manifest to ``root/manifest.json``."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)

    def _next_id(self) -> str:
        """Return the next sequential version id (``"T-0001"`` style)."""
        highest = 0
        for entry in self._manifest["versions"]:
            try:
                n = int(str(entry["id"]).split("-", 1)[1])
            except (IndexError, KeyError, ValueError):
                continue
            highest = max(highest, n)
        return f"T-{highest + 1:04d}"

    def _entry(self, version_id: str) -> dict:
        """Return the manifest entry for ``version_id`` or raise ``KeyError``."""
        for entry in self._manifest["versions"]:
            if entry["id"] == version_id:
                return entry
        raise KeyError(f"unknown version id: {version_id!r}")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def commit(self, label: str, payload: dict) -> str:
        """Store ``payload`` as the next T-numbered snapshot.

        The payload is serialized canonically
        (``json.dumps(payload, sort_keys=True, separators=(",", ":"))``),
        written to ``snapshots/{id}.json`` as utf-8, and its sha256 (of the
        exact bytes written) is recorded in the manifest.

        Args:
            label: Human-readable label for this snapshot.
            payload: JSON-serializable dict to archive.

        Returns:
            The new version id, e.g. ``"T-0001"``.
        """
        version_id = self._next_id()
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        raw = data.encode("utf-8")
        snapshot_path = self.snapshots_dir / f"{version_id}.json"
        with open(snapshot_path, "wb") as f:
            f.write(raw)
        entry = {
            "id": version_id,
            "label": label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "path": f"snapshots/{version_id}.json",
        }
        self._manifest["versions"].append(entry)
        self._write_manifest()
        return version_id

    def restore(self, version_id: str) -> dict:
        """Load and verify the payload stored under ``version_id``.

        Reads the snapshot bytes, recomputes their sha256, and compares it to
        the manifest record.

        Args:
            version_id: A version id previously returned by :meth:`commit`.

        Returns:
            The parsed payload dict.

        Raises:
            KeyError: If ``version_id`` is not in the manifest.
            IntegrityError: If the snapshot bytes do not match the recorded
                sha256 (i.e. the file was tampered with).
        """
        entry = self._entry(version_id)
        snapshot_path = self.root / entry["path"]
        with open(snapshot_path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise IntegrityError(
                f"snapshot {version_id} failed integrity check: "
                f"expected sha256 {entry['sha256']}, got {digest}"
            )
        return json.loads(raw.decode("utf-8"))

    def versions(self) -> list[dict]:
        """Return manifest entries, oldest first."""
        return list(self._manifest["versions"])

    def latest(self) -> str | None:
        """Return the most recent version id, or ``None`` if the archive is empty."""
        if not self._manifest["versions"]:
            return None
        return self._manifest["versions"][-1]["id"]

    def diff(self, a: str, b: str) -> dict:
        """Compare the top-level keys of two archived payloads.

        Args:
            a: Version id of the older/base payload.
            b: Version id of the newer payload.

        Returns:
            ``{"added": [...], "removed": [...], "changed": [...]}`` where
            *added* keys exist only in ``b``, *removed* keys exist only in
            ``a``, and *changed* keys exist in both with unequal values.
            Each list is sorted.
        """
        payload_a = self.restore(a)
        payload_b = self.restore(b)
        keys_a = set(payload_a)
        keys_b = set(payload_b)
        return {
            "added": sorted(keys_b - keys_a),
            "removed": sorted(keys_a - keys_b),
            "changed": sorted(
                k for k in keys_a & keys_b if payload_a[k] != payload_b[k]
            ),
        }
