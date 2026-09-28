"""What a call actually did to memory - counted where it happens.

§11.122. The retrieval engine is meant to say what a retrieval cost, and
its first honest bill wrapped the memory in a counting proxy for the
length of one call. GPT-6 Astra's adversarial review took that apart:
two threads sharing an engine corrupted each other's bills and left the
proxy installed; a suggester that checked the memory's type lost its
answer; every index query the suggester made was billed as a phrase
probe; and nested calls were charged to enclosing bills for reads but
not for semantic calls.

So the memory is never substituted. A call that wants a bill opens one
with :func:`metering`, and the memory's own ``recall_fact`` and
``find_facts`` charge every bill open in the current context. Context
is a :mod:`contextvars` stack, so threads (and tasks) never see each
other's bills, and a call nested inside another is charged to both -
one policy, for every counter. Phrase depth is raised only by the code
that IS the phrase path, through :func:`phrase_path`.

**A bill is exact, or it says it is not.** Two more reviews found work
escaping the context and the bill reporting itself complete anyway:
delegated to a thread pool; delegated to a thread that opened a bill of
its own; done by a nested retrieval over a memory that does not meter;
still running when the call returned. Patching each route in turn was
the wrong shape, so the guarantee is made once instead:

* every metered memory operation, anywhere in the process, advances one
  counter - O(1), and only while some bill is open;
* a bill records that counter when it opens and when it closes, and
  anything that happened in the window which was not charged to it is
  counted as ``unattributed`` - an upper bound on missed work, which
  may include unrelated work on other threads;
* :func:`carry` hands the caller's bills to delegated work, and a bill
  that closes while carried work is still running is incomplete;
* :func:`mark_enclosing_incomplete` lets a call that cannot meter itself
  taint every bill that encloses it.

A bill is **complete** only if none of those fired. The one thing no
in-process meter can see is work queued but not yet started when the
call returns: a suggester must not abandon work it submitted.

With no bill open anywhere, each hook costs one context-variable read
and one integer truth test.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["Bill", "carry", "charge_index", "charge_lookup",
           "charge_semantic", "mark_enclosing_incomplete", "metering",
           "phrase_path"]


@dataclass
class Bill:
    """What one metered call did. Per call, never cumulative."""

    lookups: int = 0
    index: int = 0
    phrase: int = 0
    semantic: int = 0
    #: Operations in this bill's window it was not charged for.
    unattributed: int = 0
    #: Set when something the bill cannot see touched its call.
    incomplete: bool = False
    #: Carried invocations running now, and at the moment of closing.
    in_flight: int = 0
    in_flight_at_close: int = 0
    _opened_at: int = 0

    @property
    def complete(self) -> bool:
        return (not self.incomplete and self.unattributed == 0
                and self.in_flight_at_close == 0)


_BILLS: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_bills", default=())
_PHRASE: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_phrase_depth", default=0)
_LOCK = threading.Lock()
#: [bills open anywhere, metered operations while any was open]
_STATE = [0, 0]


@contextmanager
def metering(bill: Bill):
    """Charge ``bill`` for everything done to memory inside the block."""
    token = _BILLS.set(_BILLS.get() + (bill,))
    with _LOCK:
        _STATE[0] += 1
        bill._opened_at = _STATE[1]
    try:
        yield bill
    finally:
        _BILLS.reset(token)
        with _LOCK:
            window = _STATE[1] - bill._opened_at
            _STATE[0] -= 1
            bill.in_flight_at_close = bill.in_flight
        bill.unattributed = max(0, window - bill.lookups - bill.index)


def carry(fn):
    """``fn``, run with this context's open bills wherever it runs.

    For work a suggester hands to a thread or an executor. Each call gets
    its own copy of the context, so the same carried function can run on
    several workers at once; the bills are shared, charged under a lock,
    and told how many carried calls are still running.
    """
    base = contextvars.copy_context()
    bills = base.get(_BILLS, ())

    @functools.wraps(fn)
    def carried(*args, **kwargs):
        with _LOCK:
            for bill in bills:
                bill.in_flight += 1
        try:
            return base.copy().run(fn, *args, **kwargs)
        finally:
            with _LOCK:
                for bill in bills:
                    bill.in_flight -= 1

    return carried


@contextmanager
def phrase_path():
    """Mark the block as phrase probing - `_reach`, `_reachable_facts`."""
    token = _PHRASE.set(_PHRASE.get() + 1)
    try:
        yield
    finally:
        _PHRASE.reset(token)


def mark_enclosing_incomplete(own: Bill | None = None) -> None:
    """Every bill open in this context, except ``own``, is now unsure."""
    with _LOCK:
        for bill in _BILLS.get():
            if bill is not own:
                bill.incomplete = True


def _charge(field: str) -> None:
    if not _STATE[0]:
        return                    # nothing open anywhere: the fast path
    bills = _BILLS.get()
    phrase = field == "index" and _PHRASE.get() > 0
    with _LOCK:
        _STATE[1] += 1
        for bill in bills:
            setattr(bill, field, getattr(bill, field) + 1)
            if phrase:
                bill.phrase += 1


def charge_lookup() -> None:
    """One fact read, charged to every bill open in this context."""
    _charge("lookups")


def charge_index() -> None:
    """One index query - a phrase probe too, if inside the phrase path."""
    _charge("index")


def charge_semantic() -> None:
    """One call to a suggester, charged to every bill open in context."""
    bills = _BILLS.get()
    if bills:
        with _LOCK:
            for bill in bills:
                bill.semantic += 1
