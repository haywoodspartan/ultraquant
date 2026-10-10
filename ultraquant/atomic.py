"""Write a file so that a crash leaves the previous one.

Every store this package keeps on disk was once written in place:
``open(path, "w")`` truncates the file first, so a process killed between
the truncation and the last byte leaves a file shorter than it should be,
and the next start fails to parse it. The settings file, the stash, the
vault catalog and the approver journal each solved this on their own; the
fact store, the router table, the archive manifest and its snapshots,
packed libraries, the dispatch experience and the distillation ledger did
not (§11.176). A truncated ``memory.json`` stopped the GUI, the TUI and the
chat from starting at all, and a crash during the second compaction of a
library destroyed every shard packed into it.

This module is the one place that knows the pattern: write to a temporary
file in the **same directory** (so the final rename never crosses a
filesystem), flush and fsync it, then ``os.replace`` it over the
destination. The destination therefore holds either the previous complete
file or the new complete file, never a mixture. A failure at any step
unlinks the temporary file and re-raises; nothing is swallowed.

:func:`write_json` serialises before opening anything, so an object that
cannot be serialised raises without creating a file.

The bytes written are the same bytes the in-place writers wrote: text mode
keeps the platform newline translation the old ``open(path, "w")`` had, so a
store rewritten through here is byte-identical to one written before.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import IO, Any, Iterator

__all__ = ["replacing", "write_bytes", "write_text", "write_json"]


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence of the directory entry on POSIX.

    Windows has no directory fsync; the ``os.replace`` there is already a
    single metadata operation.
    """
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextlib.contextmanager
def replacing(path: str | os.PathLike, mode: str = "wb", *,
              encoding: str | None = None) -> Iterator[IO[Any]]:
    """Yield a handle on a temporary file that replaces ``path`` on a clean exit.

    Args:
        path: The destination. Its parent directory is created if missing.
        mode: ``"wb"`` or ``"w"``; a read, append or update mode is refused
            because the temporary file starts empty.
        encoding: Text encoding for ``"w"``; UTF-8 when not given.

    Raises:
        ValueError: For a mode that is not a plain write.
        OSError: From the write or the final replace; the temporary file is
            removed first and the destination is untouched.
    """
    if any(flag in mode for flag in "ra+") or "w" not in mode:
        raise ValueError(f"replacing() writes a fresh file; mode {mode!r} is not a plain write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent),
                                    prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        try:
            if "b" in mode:
                handle = os.fdopen(fd, mode)
            else:
                handle = os.fdopen(fd, mode, encoding=encoding or "utf-8")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _fsync_directory(target.parent)


def write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Replace ``path`` with ``data``, or leave it exactly as it was."""
    with replacing(path, "wb") as handle:
        handle.write(data)


def write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text``, or leave it exactly as it was."""
    with replacing(path, "w", encoding=encoding) as handle:
        handle.write(text)


def write_json(path: str | os.PathLike, obj: Any, **dumps_kwargs: Any) -> None:
    """Serialise ``obj`` first, then replace ``path`` with the result.

    ``dumps_kwargs`` go to :func:`json.dumps` unchanged, so a writer keeps
    its ``sort_keys``, ``indent`` and ``separators`` and the file keeps its
    bytes. An unserialisable object raises :class:`TypeError` before any
    file is opened.
    """
    text = json.dumps(obj, **dumps_kwargs)
    write_text(path, text)
