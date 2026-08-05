"""Self-learning: build off what the model already has.

Two mechanisms:

* **Teaching** — a demonstrated glyph is augmented with seeded noise and used to
  *continue* training the relevant category expert from its stored weights, so
  each lesson accumulates rather than restarting.
* **Consolidation** — the shards that have actually been recalled get packed out
  of loose files into a library container, the stores are saved, and the whole
  state is committed to the Ar(T)chive.  This is the housekeeping pass that turns
  scattered recent learning into a compact, catalogued, verifiable layer.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from ultraquant.experts.moe import ExpertPool
from ultraquant.pattern.recognition import render, row_means

__all__ = ["SelfLearner"]

_FACT_PATTERNS = (
    re.compile(r"^remember(?:,|:|\s+that)?\s+(?P<key>.{1,60}?)\s*=\s*(?P<value>.+)$", re.I),
    re.compile(r"^remember(?:,|:|\s+that)?\s+(?P<key>.{1,60}?)\s+is\s+(?P<value>.+)$", re.I),
    re.compile(r"^(?P<key>[\w \-']{1,40}?)\s+is\s+(?P<value>.+)$", re.I),
)


class SelfLearner:
    """Turns interactions into durable, catalogued learning."""

    def __init__(self, session: Any) -> None:
        """Bind to a session.

        Args:
            session: The :class:`~ultraquant.interpreter.thoughts.Session`.
        """
        self.session = session

    # ------------------------------------------------------------------ facts

    @staticmethod
    def extract_facts(text: str) -> list[tuple[str, str]]:
        """Pull ``(key, value)`` statements out of free text."""
        out: list[tuple[str, str]] = []
        for line in text.splitlines():
            line = line.strip().rstrip(".")
            if not line:
                continue
            for pattern in _FACT_PATTERNS:
                match = pattern.match(line)
                if match:
                    key = match.group("key").strip().lower()
                    for article in ("the ", "a ", "an "):
                        if key.startswith(article):
                            key = key[len(article):]
                    value = match.group("value").strip()
                    if key and value:
                        out.append((key, value))
                    break
        return out

    # ----------------------------------------------------------------- glyphs

    def teach_glyph(
        self,
        category: str,
        labels: list[str],
        rows: list[str],
        label: str,
        epochs: int = 15,
        samples: int = 24,
        flips: int = 2,
    ) -> dict:
        """Teach one glyph by example, continuing from the stored expert.

        Args:
            category: Category whose expert learns this.
            labels: Full ordered label set for that expert.
            rows: Five 5-character glyph rows.
            label: The label this glyph should carry (must be in ``labels``).
            epochs: Training passes.
            samples: Noisy variants generated from the demonstration.
            flips: Pixels flipped per variant.

        Returns:
            The training report from :meth:`ExpertPool.train_expert`.
        """
        if label not in labels:
            raise ValueError(f"label {label!r} not in the expert's label set")
        pixels = render(rows)
        rng = random.Random(f"teach|{category}|{label}|{''.join(rows)}")
        target = labels.index(label)

        xs: list[list[float]] = []
        ys: list[int] = []
        for _ in range(samples):
            variant = list(pixels)
            for idx in rng.sample(range(len(variant)), flips):
                variant[idx] = 1.0 - variant[idx]
            xs.append(variant + row_means(variant))
            ys.append(target)
        # Keep the clean demonstration in the batch too.
        xs.append(pixels + row_means(pixels))
        ys.append(target)

        report = self.session.experts.train_expert(
            category, xs, ys, labels, epochs=epochs, lr=0.05
        )
        # Record what this category looks like, or the lesson is invisible to
        # pattern routing: a taught glyph would still be routed by keywords it
        # does not contain, and the expert just trained would never be reached.
        shard_id = ExpertPool.shard_id(category)
        if self.session.vault.has(shard_id):
            existing = self.session.vault.entry(shard_id).get("signature") or []
            self.session.vault.set_signature(shard_id, [*existing, pixels])
            # A long teaching run must not bloat the catalog before anyone
            # thinks to run maintenance: measured at 200 lessons, one
            # category's signature reached 202 prototypes, 25.6 KB resident
            # and 377 us per routing call. Twice the budget is the tripwire.
            from ultraquant.shards.prototypes import (
                PROTOTYPE_BUDGET, consolidate_signatures,
            )

            if len(existing) + 1 > 2 * PROTOTYPE_BUDGET:
                consolidate_signatures(self.session.vault)
        self.session.router.learn(f"{label} {category} glyph shape pattern", category, delta=0.2)
        shard_id = ExpertPool.shard_id(category)
        if self.session.vault.has(shard_id):
            self.session.vault.reinforce(shard_id, [label, category, "glyph"], delta=0.2)
        self.session.memory.remember_episode(
            "teaching",
            {"category": category, "label": label, "rows": list(rows), **report},
            tags=["teach", category],
        )
        self.session.save()
        return report

    # ---------------------------------------------------------- consolidation

    def consolidate(self, min_access: int = 2) -> dict:
        """Pack recalled shards into a library and snapshot the whole state.

        Args:
            min_access: Minimum access count for a loose shard to be packed.

        Returns:
            ``{"packed": n, "library": path|None, "snapshot": T-id|None}``.
        """
        session = self.session
        vault = session.vault

        # Episodic traces consolidate into semantic prototypes in the same
        # pass that packs hot shards - this is the sleep-like step where
        # scattered recent learning becomes compact catalogued structure.
        from ultraquant.shards.prototypes import consolidate_signatures

        prototypes = consolidate_signatures(vault)

        hot = [
            entry["shard_id"]
            for entry in vault.catalog()
            if entry["location"] == "loose" and entry["access_count"] >= min_access
        ]

        library_path: str | None = None
        packed = 0
        if hot:
            lib_dir = Path(vault.root) / "library"
            lib_dir.mkdir(parents=True, exist_ok=True)
            index = len(list(lib_dir.glob("uq_lib_*.uql"))) + 1
            target = lib_dir / f"uq_lib_{index:04d}.uql"
            packed = vault.pack(target, hot, prune_loose=True)
            library_path = str(target)

        session.save()

        snapshot: str | None = None
        if session.archive is not None:
            snapshot = session.archive.commit(
                "consolidation",
                {
                    "vault": vault.stats(),
                    "catalog": [
                        {
                            "shard_id": e["shard_id"],
                            "category": e["category"],
                            "location": e["location"],
                            "nbytes": e["nbytes"],
                            "access_count": e["access_count"],
                        }
                        for e in vault.catalog()
                    ],
                    "router": session.router.state(),
                    "memory": session.memory.stats(),
                    "cache": session.cache.stats(),
                    "stash": session.stash.stats(),
                },
            )

        session.memory.remember_episode(
            "consolidation",
            {"packed": packed, "library": library_path, "snapshot": snapshot},
            tags=["consolidate"],
        )
        session.memory.save()
        return {"packed": packed, "library": library_path, "snapshot": snapshot,
                "prototypes": prototypes}
