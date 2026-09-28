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

A fourth review combined those routes and got past each safeguard, so
everything a bill cannot see moves a process-wide signal it samples at
open and close - memory operations, suggester calls, and calls over
memories that do not meter - and a charge that arrives from a thread
other than the bill's own, outside a :func:`carry` invocation, marks
the bill incomplete (a hand-copied context carries charges but no
lifetime).

**Where an operation happens.** The fifth review showed that "began
before the call returned" cannot be observed: an operation paused one
instruction before registering is invisible to any snapshot, and moving
the snapshot later only moves the gap. So an operation counts at its
**registration point** - its entry hook, under the same lock the bill's
open and close snapshots take. What a bill promises is then exact and
linearizable, and it is this:

* **complete** means every operation registered while the bill was open
  was charged to it; no work delegated through :func:`carry` was still
  outstanding - queued or running - when it closed; and no charge from
  another thread outside carry(), no unmetered memory, and no suggester
  that does not meter itself registered in its window;
* **out of contract** is work that registers nothing before the bill
  closes and was not delegated through carry(). No in-process meter can
  observe it; a bill says nothing about it.

carry() registers its work when it WRAPS the function, not when a worker
starts it, and a carried function runs once - so work queued behind a
busy pool is outstanding at close, and the bill knows.

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
    #: Carried calls registered and not yet finished - queued or running -
    #: now, and at the moment of closing.
    outstanding: int = 0
    outstanding_at_close: int = 0
    _opened_at: tuple = (0, 0, 0)
    _owner: int = 0
    #: Set, with everything above finalized, under the lock that closes
    #: the bill. A closed bill is frozen: a context copied while it was
    #: open still holds it, and nothing charged through that copy may
    #: move it (the seventh review: a late charge erased a real gap).
    _closed: bool = False

    @property
    def complete(self) -> bool:
        return (not self.incomplete and self.unattributed == 0
                and self.outstanding_at_close == 0)


_BILLS: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_bills", default=())
_PHRASE: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_phrase_depth", default=0)
#: The carried invocation a context belongs to - matched against the
#: thread actually running it, because a context is copied into any
#: thread that asks, and an exemption that travelled with it would let an
#: unregistered child thread charge as if carried (the sixth review).
_CARRIED: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_carried", default=None)
_EXECUTING = threading.local()
_LOCK = threading.Lock()
#: [bills open anywhere, then - counted while any bill is open - metered
#: memory operations, suggester calls, unmetered calls]
_STATE = [0, 0, 0, 0]


@contextmanager
def metering(bill: Bill):
    """Charge ``bill`` for everything done to memory inside the block."""
    token = _BILLS.set(_BILLS.get() + (bill,))
    with _LOCK:
        _STATE[0] += 1
        bill._opened_at = (_STATE[1], _STATE[2], _STATE[3])
        bill._owner = threading.get_ident()
    try:
        yield bill
    finally:
        _BILLS.reset(token)
        with _LOCK:
            memory_ops = _STATE[1] - bill._opened_at[0]
            suggester_calls = _STATE[2] - bill._opened_at[1]
            unmetered_calls = _STATE[3] - bill._opened_at[2]
            _STATE[0] -= 1
            bill.outstanding_at_close = bill.outstanding
            bill.unattributed = max(0, memory_ops - bill.lookups
                                    - bill.index)
            if suggester_calls > bill.semantic or unmetered_calls:
                bill.incomplete = True
            bill._closed = True


def carry(fn):
    """``fn``, to run ONCE elsewhere, charged to this context's open bills.

    For work a suggester hands to a thread or an executor. The work is
    registered on every open bill here, when it is wrapped - so a bill
    that closes while it is queued or running knows it is outstanding -
    and released when the one call finishes. Carry each call separately:
    a second call raises, because its lifetime could not be observed.
    """
    base = contextvars.copy_context()
    with _LOCK:
        bills = tuple(bill for bill in base.get(_BILLS, ())
                      if not bill._closed)
        for bill in bills:
            bill.outstanding += 1
    state = {"used": False}

    def inside(*args, **kwargs):
        token = object()
        _CARRIED.set(token)
        previous = getattr(_EXECUTING, "token", None)
        _EXECUTING.token = token
        try:
            return fn(*args, **kwargs)
        finally:
            _EXECUTING.token = previous

    @functools.wraps(fn)
    def carried(*args, **kwargs):
        with _LOCK:
            if state["used"]:
                raise RuntimeError("a carried function runs once - carry() "
                                   "each call, so its lifetime is observed")
            state["used"] = True
        try:
            return base.copy().run(inside, *args, **kwargs)
        finally:
            with _LOCK:
                for bill in bills:
                    bill.outstanding -= 1

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
    """A call that cannot meter itself: every bill that may enclose it is
    unsure - those in this context directly, any other through the
    process-wide unmetered count it samples."""
    with _LOCK:
        if _STATE[0]:
            _STATE[3] += 1
        for bill in _BILLS.get():
            if bill is not own and not bill._closed:
                bill.incomplete = True


def _foreign(bills: tuple) -> None:
    """Charges from another thread, outside carry(): no lifetime known.

    Called with the lock held."""
    token = _CARRIED.get()
    if token is not None and getattr(_EXECUTING, "token", None) is token:
        return                    # this thread IS the carried invocation
    me = threading.get_ident()
    for bill in bills:
        if bill._owner != me and not bill._closed:
            bill.incomplete = True


def _charge(field: str) -> None:
    if not _STATE[0]:
        return                    # nothing open anywhere: the fast path
    bills = _BILLS.get()
    phrase = field == "index" and _PHRASE.get() > 0
    with _LOCK:
        _STATE[1] += 1
        _foreign(bills)
        for bill in bills:
            if bill._closed:
                continue
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
    if not _STATE[0]:
        return
    bills = _BILLS.get()
    with _LOCK:
        _STATE[2] += 1
        _foreign(bills)
        for bill in bills:
            if not bill._closed:
                bill.semantic += 1
