"""Pattern recognition decides which shards to read.

The model is the library, and the library stays on storage. So the question at
inference time is never "load the model" — it is *which few shards does this
input actually need?* Pattern recognition answers it: the input is recognised,
the recognition names categories, the categories name shards, and only those
shards are pulled off NVMe into RAM.

That is what keeps a trillion-parameter store usable on a small machine. Nothing
here scales with the size of the model; it scales with the size of the answer.

:class:`PatternWorkingSet` does three things:

* **Predict** — turn an input into the ranked shard ids likely to be needed,
  using the router's associative weights (which recall has been reinforcing all
  along) plus recognised pattern labels.
* **Prefetch** — pull those shards into the RAM tier before they are asked for,
  optionally pinning them so ordinary traffic cannot evict the working set.
* **Account** — report how much of the library was resident, which is the number
  that makes the claim checkable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ultraquant.experts.moe import ExpertPool

__all__ = ["Prediction", "WorkingSetReport", "PatternWorkingSet"]


@dataclass(frozen=True)
class Prediction:
    """A shard the recogniser expects to need, and why."""

    shard_id: str
    category: str
    score: float
    reason: str


@dataclass
class WorkingSetReport:
    """What a prefetch did."""

    predicted: list[Prediction] = field(default_factory=list)
    loaded: int = 0
    already_resident: int = 0
    missing: list[str] = field(default_factory=list)
    bytes_loaded: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{len(self.predicted)} shard(s) predicted, {self.loaded} paged in, "
            f"{self.already_resident} already hot, {self.bytes_loaded:,} bytes "
            f"in {self.seconds * 1000:.1f} ms"
        )


class PatternWorkingSet:
    """Keeps the shards pattern recognition needs resident, and nothing else."""

    def __init__(
        self,
        vault,
        router,
        tier=None,
        recognizer=None,
        top_k: int = 3,
    ) -> None:
        """Bind to a library and its RAM tier.

        Args:
            vault: The :class:`~ultraquant.shards.vault.ShardVault`.
            router: The :class:`~ultraquant.shards.router.CategoryRouter`.
            tier: A :class:`~ultraquant.storage.tiered.TieredStorage` to pin
                into. If None, the vault's storage is used when it is tiered,
                otherwise prediction still works and prefetch is a no-op.
            recognizer: Optional pattern recognizer for glyph inputs.
            top_k: How many categories to consider per input.
        """
        self.vault = vault
        self.router = router
        self.recognizer = recognizer
        self.top_k = int(top_k)
        self.tier = tier if tier is not None else self._tier_of(vault)
        self.pinned: set[str] = set()

    @staticmethod
    def _tier_of(vault):
        """The vault's RAM tier, if it has one."""
        storage = getattr(vault, "storage", None)
        return storage if hasattr(storage, "pin") else None

    # -- prediction --------------------------------------------------------

    def predict(self, text: str = "", labels: list[str] | None = None) -> list[Prediction]:
        """Rank the shards this input is likely to need.

        Two signals combine: the router's category ranking (built from keyword
        overlap *and* the association weights that every past recall has
        strengthened), and any pattern labels already recognised.

        Args:
            text: The input, if it is textual.
            labels: Pattern labels already recognised, if any.

        Returns:
            Predictions, best first.
        """
        scores: dict[str, tuple[float, str, str]] = {}

        for category, score in (self.router.route(text, top_k=self.top_k) if text else []):
            shard_id = ExpertPool.shard_id(category)
            scores[shard_id] = (float(score), category, "routed by association")

        for label in labels or []:
            for entry in self.vault.catalog():
                if label in entry.get("associations", {}) or label == entry["category"]:
                    prior = scores.get(entry["shard_id"], (0.0, entry["category"], ""))
                    weight = float(entry.get("associations", {}).get(label, 0.5))
                    scores[entry["shard_id"]] = (
                        prior[0] + weight, entry["category"], f"recognised {label!r}"
                    )

        known = {entry["shard_id"] for entry in self.vault.catalog()}
        ranked = [
            Prediction(shard_id, category, score, reason)
            for shard_id, (score, category, reason) in scores.items()
            if shard_id in known
        ]
        ranked.sort(key=lambda p: (-p.score, p.shard_id))
        return ranked

    # -- prefetch ----------------------------------------------------------

    def prefetch(
        self,
        text: str = "",
        labels: list[str] | None = None,
        pin: bool = False,
        limit: int | None = None,
    ) -> WorkingSetReport:
        """Predict, then pull the predicted shards into RAM.

        Args:
            text: The input, if textual.
            labels: Recognised pattern labels, if any.
            pin: Hold the shards against eviction.
            limit: Cap on shards to fetch.

        Returns:
            A :class:`WorkingSetReport`.
        """
        started = time.perf_counter()
        predicted = self.predict(text, labels)
        if limit is not None:
            predicted = predicted[:limit]

        report = WorkingSetReport(predicted=predicted)
        if self.tier is None or not predicted:
            report.seconds = time.perf_counter() - started
            return report

        result = self.tier.prefetch_ranges(self._ranges(predicted), pin=pin)
        report.loaded = result["loaded"]
        report.already_resident = result["already_resident"]
        report.bytes_loaded = result["bytes"]
        report.missing = result["missing"]
        if pin:
            self.pinned.update(p.shard_id for p in predicted)
        report.seconds = time.perf_counter() - started
        return report

    def _ranges(self, predicted: list[Prediction]) -> list[tuple[str, int, int]]:
        """Turn predicted shards into the byte ranges that hold them.

        A packed shard is a range inside a library that may be far larger than
        memory, so the working set is expressed as ranges — never as objects.
        """
        ranges: list[tuple[str, int, int]] = []
        for prediction in predicted:
            try:
                entry = self.vault.entry(prediction.shard_id)
            except KeyError:
                continue
            key = self.vault.storage_key(prediction.shard_id)
            if entry["location"] == "loose":
                ranges.append((key, 0, int(entry["nbytes"])))
            else:
                ranges.append((key, int(entry["offset"]), int(entry["length"])))
        return ranges

    def pin_categories(self, categories: list[str]) -> WorkingSetReport:
        """Pin whole categories — the deliberate 'keep this hot' control."""
        started = time.perf_counter()
        predicted = [
            Prediction(ExpertPool.shard_id(c), c, 1.0, "pinned by name")
            for c in categories
            if self.vault.has(ExpertPool.shard_id(c))
        ]
        report = WorkingSetReport(predicted=predicted)
        if self.tier is None or not predicted:
            report.seconds = time.perf_counter() - started
            return report
        result = self.tier.prefetch_ranges(self._ranges(predicted), pin=True)
        report.loaded = result["loaded"]
        report.already_resident = result["already_resident"]
        report.bytes_loaded = result["bytes"]
        report.missing = result["missing"]
        self.pinned.update(p.shard_id for p in predicted)
        report.seconds = time.perf_counter() - started
        return report

    def release(self) -> None:
        """Unpin everything this working set pinned."""
        if self.tier is None:
            return
        for shard_id in list(self.pinned):
            try:
                entry = self.vault.entry(shard_id)
            except KeyError:
                continue
            key = self.vault.storage_key(shard_id)
            if entry["location"] == "loose":
                self.tier.unpin(key)
            else:
                self.tier.unpin_range(key, int(entry["offset"]), int(entry["length"]))
        self.pinned.clear()

    # -- accounting --------------------------------------------------------

    def stats(self) -> dict:
        """Residency and hit-rate figures, with the fraction that matters."""
        if self.tier is None:
            return {"tiered": False, "pinned_keys": 0}
        info = dict(self.tier.tier_stats())
        info["tiered"] = True
        info["working_set_keys"] = len(self.pinned)
        return info
