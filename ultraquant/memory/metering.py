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

With no bill open, each hook costs one context-variable read.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["Bill", "charge_index", "charge_lookup", "charge_semantic",
           "metering", "phrase_path"]


@dataclass
class Bill:
    """What one metered call did. Per call, never cumulative."""

    lookups: int = 0
    index: int = 0
    phrase: int = 0
    semantic: int = 0


_BILLS: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_bills", default=())
_PHRASE: contextvars.ContextVar = contextvars.ContextVar(
    "ultraquant_phrase_depth", default=0)


@contextmanager
def metering(bill: Bill):
    """Charge ``bill`` for everything done to memory inside the block."""
    token = _BILLS.set(_BILLS.get() + (bill,))
    try:
        yield bill
    finally:
        _BILLS.reset(token)


@contextmanager
def phrase_path():
    """Mark the block as phrase probing - `_reach`, `_reachable_facts`."""
    token = _PHRASE.set(_PHRASE.get() + 1)
    try:
        yield
    finally:
        _PHRASE.reset(token)


def charge_lookup() -> None:
    """One fact read, charged to every open bill."""
    for bill in _BILLS.get():
        bill.lookups += 1


def charge_index() -> None:
    """One index query - a phrase probe too, if inside the phrase path."""
    bills = _BILLS.get()
    if not bills:
        return
    phrase = _PHRASE.get() > 0
    for bill in bills:
        bill.index += 1
        if phrase:
            bill.phrase += 1


def charge_semantic() -> None:
    """One call to a suggester, charged to every open bill."""
    for bill in _BILLS.get():
        bill.semantic += 1
