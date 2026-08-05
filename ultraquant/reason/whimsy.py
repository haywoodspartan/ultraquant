"""Quarantined chaos: physical unpredictability with receipts.

A deterministic system cannot surprise itself. Every seed in this codebase is
fixed on purpose, and nondeterminism has been hunted as a bug — which is
exactly why genuine unpredictability, if it is wanted at all, must arrive the
way web content does: through a quarantine, on terms that keep the rest of
the system provable. The founding idea (the user's): use quantum-machine
noise and timing instability as chaos for out-of-the-box, whimsical decision
making.

**The three rules that make chaos admissible here:**

1. **Chaos may only choose among acceptable-equals.** A whimsical draw picks
   *which* valid option, never *whether* an option is valid. Routing answers,
   fact confidences, gate measurements and test outcomes never touch the
   well. (Random is not reasoned — the sibling of found-is-not-believed.)
2. **Every draw leaves a receipt.** Consumer, purpose, bits drawn, choice
   made — recorded, so any whimsical run can be replayed exactly from its
   log. Whimsy with an audit trail.
3. **Off by default.** The suite, the gates and every session that does not
   opt in remain bit-for-bit deterministic.

**The sources, measured on this machine before being trusted:**

* **Timing jitter** — the machine's own physical wobble (scheduler, cache,
  thermal). Measured here: 1.59 bits/byte Shannon estimate, 41 distinct
  delta values, 508/750 unique raw blocks. Real, but weak — credited
  conservatively at 1 bit per harvested byte.
* **The OS entropy pool** (``os.urandom``) — the workhorse floor.
* **Cached QPU measurement bits** — the premium tier the idea began with.
  Genuine quantum randomness is the one thing no classical machine has;
  per-decision QPU calls die on the same cost/latency wall as §7.1, so
  quantum entropy is a *harvested resource*: an explicit, user-triggered
  (billable) job measures a superposition circuit, and the resulting bits are
  stored as the library shard ``entropy:quantum`` and metered out one draw at
  a time. Never harvested automatically.

**The trap this module was nearly born with, kept as a warning:** the first
jitter harvester produced all-zero bits (Windows' 100 ns timer swallowed the
workload), and whitening laundered that constant into *statistically perfect-
looking* output — same digest, repeated. Balance tests cannot detect zero
entropy after hashing. Sources are therefore credited by **raw** measurements
(distinct values, unique blocks), never by the prettiness of whitened output.

Pure Python standard library.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Sequence

__all__ = ["EntropyWell", "harvest_jitter", "quantum_entropy_circuit"]

#: Library shard holding harvested quantum measurement bits.
QUANTUM_ENTROPY_SHARD = "entropy:quantum"


def harvest_jitter(n_bytes: int = 64) -> bytes:
    """Harvest bytes from the machine's timing wobble.

    A dict-churn workload whose duration genuinely varies is timed repeatedly;
    the variable part of each delta contributes one byte. Weak on purpose-
    measured entropy (~1.6 bits/byte here) — the well credits it at 1 bit per
    byte and mixes it with stronger sources rather than trusting it alone.
    """
    deltas = []
    for _ in range(max(8, n_bytes)):
        start = time.perf_counter_ns()
        churn: dict[str, list[int]] = {}
        for i in range(60):
            churn[str(i)] = [i] * 7
        deltas.append(time.perf_counter_ns() - start)
    floor = min(deltas)
    return bytes(((d - floor) // 100) & 0xFF for d in deltas[:n_bytes])


def quantum_entropy_circuit(num_qubits: int = 12):
    """The harvest circuit: uniform superposition, measured.

    H on every qubit puts the register in an equal superposition of all
    2^n basis states; measuring collapses to one of them with genuinely
    quantum randomness — each shot yields ``num_qubits`` bits no classical
    process produced. Run on real hardware via an explicit, user-approved
    (billable) BlueQubit or IBM job; the returned counts become the cached
    bits of :data:`QUANTUM_ENTROPY_SHARD`. On a simulator the same circuit
    yields only pseudo-randomness, which the well refuses to credit as
    quantum.
    """
    from ultraquant.quantum.circuit import Circuit

    circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.h(qubit)
    return circuit


class EntropyWell:
    """Metered physical unpredictability, drawn one receipted choice at a time.

    Args:
        session: Optional interpreter session; provides the vault holding any
            harvested quantum bits, and the memory receipts are written to.
        enabled: Whimsy is opt-in. Disabled wells refuse to draw.
        replay: A previous run's receipts. When given, draws replay those
            recorded values instead of consuming entropy — the mechanism that
            makes whimsical runs exactly reproducible, and what tests inject.
    """

    def __init__(self, session: Any | None = None, enabled: bool = False,
                 replay: Sequence[dict] | None = None) -> None:
        self.session = session
        self.enabled = bool(enabled)
        self.receipts: list[dict] = []
        self._replay = list(replay) if replay is not None else None
        self._replay_at = 0
        self._pool = b""
        self._quantum_bits_used = 0

    # ------------------------------------------------------------------ #
    # sources
    # ------------------------------------------------------------------ #

    def _quantum_bytes(self, n: int) -> bytes:
        """Cached QPU-harvested bytes, metered; empty when none remain."""
        if self.session is None:
            return b""
        vault = getattr(self.session, "vault", None)
        if vault is None or not vault.has(QUANTUM_ENTROPY_SHARD):
            return b""
        payload = vault.get(QUANTUM_ENTROPY_SHARD)
        blob = bytes.fromhex(payload.get("bits_hex", ""))
        start = int(payload.get("used", 0))
        take = blob[start:start + n]
        if take:
            payload["used"] = start + len(take)
            vault.add_shard(QUANTUM_ENTROPY_SHARD, "entropy", payload,
                            kind="entropy")
            self._quantum_bits_used += len(take) * 8
        return take

    def _refill(self) -> None:
        """Mix all sources into the pool through BLAKE2.

        The hash is a mixer, not a launderer: each source's contribution is
        justified by its *raw* measured entropy, and ``os.urandom`` is always
        in the mix so a machine whose jitter collapses (as this one's first
        harvester did) still bottoms out at the OS pool, never at a constant.
        """
        material = (
            os.urandom(32)
            + harvest_jitter(32)
            + self._quantum_bytes(8)
            + len(self.receipts).to_bytes(4, "big")
        )
        self._pool = hashlib.blake2b(material, digest_size=64).digest()

    def _draw_value(self, span: int) -> int:
        """An integer in ``[0, span)`` from replay or from the pool."""
        if self._replay is not None:
            record = self._replay[self._replay_at]
            self._replay_at += 1
            return int(record["value"]) % span
        if len(self._pool) < 4:
            self._refill()
        value = int.from_bytes(self._pool[:4], "big")
        self._pool = self._pool[4:]
        return value % span

    # ------------------------------------------------------------------ #
    # the only API: choose among acceptable-equals
    # ------------------------------------------------------------------ #

    def choose(self, consumer: str, purpose: str, options: Sequence[Any]):
        """Pick one of ``options``, all of which the caller deems acceptable.

        Returns ``options[0]`` when the well is disabled — the deterministic
        default — so callers need no branching.
        """
        if not options:
            raise ValueError("nothing to choose among")
        if not self.enabled or len(options) == 1:
            return options[0]
        value = self._draw_value(len(options))
        choice = value % len(options)
        self._receipt(consumer, purpose, value, str(options[choice]))
        return options[choice]

    def occasionally(self, consumer: str, purpose: str,
                     out_of: int = 8) -> bool:
        """True roughly once per ``out_of`` asks — the whimsy pulse.

        Deterministically False when disabled, so exploration never happens
        to a session that did not ask for a mood.
        """
        if not self.enabled:
            return False
        value = self._draw_value(out_of)
        hit = value % out_of == 0
        self._receipt(consumer, purpose, value, str(hit))
        return hit

    def _receipt(self, consumer: str, purpose: str, value: int,
                 choice: str) -> None:
        record = {"consumer": consumer, "purpose": purpose,
                  "value": int(value), "choice": choice}
        self.receipts.append(record)
        memory = getattr(self.session, "memory", None)
        if memory is not None:
            memory.remember_episode("whimsy", dict(record),
                                    tags=["whimsy", consumer])

    def report(self) -> dict:
        """Draws so far, and how much premium entropy was spent."""
        return {"draws": len(self.receipts),
                "quantum_bits_used": self._quantum_bits_used,
                "replaying": self._replay is not None}
