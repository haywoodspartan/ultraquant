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

**Work that leaves the context.** The second review found memory work a
suggester handed to a thread pool vanishing from the bill: worker
threads do not inherit context variables. :func:`carry` hands the
caller's open bills to such work. Work that is NOT carried cannot be
attributed - but it is not hidden either: memory work done outside any
bill while a bill is open anywhere is counted on every open bill as
``unattributed``, so a bill that may be short says so. That count can
include unrelated work on other threads; it is a warning, never a
charge.

With no bill open anywhere, each hook costs one context-variable read
and one dictionary truth test.
"""

from __future__ import annotations

import contextvars
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["Bill", "carry", "charge_index", "charge_lookup",
           "charge_semantic", "metering", "phrase_path"]


@dataclass
class Bill:
    """What one metered call did. Per call, never cumulative."""

    lookups: int = 0
    index: int = 0
    phrase: int = 0
    semantic: int = 0
    unattributed: int = 0


_BILLS: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_bills", default=())
_PHRASE: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_phrase_depth", default=0)
#: Every bill open anywhere in the process, for work no context carries.
_OPEN: dict = {}
_LOCK = threading.Lock()


@contextmanager
def metering(bill: Bill):
    """Charge ``bill`` for everything done to memory inside the block."""
    token = _BILLS.set(_BILLS.get() + (bill,))
    with _LOCK:
        _OPEN[id(bill)] = bill
    try:
        yield bill
    finally:
        _BILLS.reset(token)
        with _LOCK:
            _OPEN.pop(id(bill), None)


def carry(fn):
    """``fn``, run with this context's open bills wherever it runs.

    For work a suggester hands to a thread or an executor. Each call gets
    its own copy of the context, so the same carried function can run on
    several workers at once; the bills themselves are shared, and charged
    under a lock.
    """
    base = contextvars.copy_context()

    @functools.wraps(fn)
    def carried(*args, **kwargs):
        return base.copy().run(fn, *args, **kwargs)

    return carried


@contextmanager
def phrase_path():
    """Mark the block as phrase probing - `_reach`, `_reachable_facts`."""
    token = _PHRASE.set(_PHRASE.get() + 1)
    try:
        yield
    finally:
        _PHRASE.reset(token)


def _unattributed() -> None:
    """Memory work outside every bill, while some bill is open."""
    if _OPEN:
        with _LOCK:
            for bill in _OPEN.values():
                bill.unattributed += 1


def charge_lookup() -> None:
    """One fact read, charged to every bill open in this context."""
    bills = _BILLS.get()
    if not bills:
        _unattributed()
        return
    with _LOCK:
        for bill in bills:
            bill.lookups += 1


def charge_index() -> None:
    """One index query - a phrase probe too, if inside the phrase path."""
    bills = _BILLS.get()
    if not bills:
        _unattributed()
        return
    phrase = _PHRASE.get() > 0
    with _LOCK:
        for bill in bills:
            bill.index += 1
            if phrase:
                bill.phrase += 1


def charge_semantic() -> None:
    """One call to a suggester, charged to every bill open in context."""
    bills = _BILLS.get()
    if bills:
        with _LOCK:
            for bill in bills:
                bill.semantic += 1
