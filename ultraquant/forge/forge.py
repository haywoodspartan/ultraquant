"""The forge: grow a complete model from scratch, as a catalogued shard library.

End to end, in one pass:

1. **Patterns** — each category's glyph prototypes are augmented into a training
   corpus (:mod:`ultraquant.forge.corpus`).
2. **Training** — every category's expert is trained *concurrently* on the
   fastest available tier (CUDA → C++ → pure Python).
3. **Shards** — each trained expert is written into the vault in the payload
   format :class:`~ultraquant.experts.moe.ExpertPool` reads, so the fresh model
   is immediately usable through the normal on-demand paging path.
4. **Routing** — category keywords are registered so text can find its expert.
5. **Library** — every shard is packed into one ``.uql`` container with an offset
   index, ready to be attached and paged in chunk by chunk.
6. **Ar(T)chive** — the whole build is snapshotted, digest and all.

The result is a model that never has to be loaded whole.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.archive.artchive import ArTchive
from ultraquant.experts.moe import ExpertPool
from ultraquant.forge.corpus import CategoryCorpus
from ultraquant.forge.trainer import ExpertSpec, TrainedExpert, train_experts
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardVault

__all__ = ["ForgeReport", "ModelForge"]

_ACT_BITS = 8


@dataclass
class ForgeReport:
    """Outcome of one forge run."""

    tier: str
    categories: int
    experts: int
    parameters: int
    train_seconds: float
    total_seconds: float
    mean_accuracy: float
    library: str | None = None
    snapshot: str | None = None
    store_bytes: int = 0
    per_category: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{self.experts} experts / {self.parameters:,} parameters "
            f"trained on '{self.tier}' in {self.train_seconds:.2f}s "
            f"(mean accuracy {self.mean_accuracy:.3f})"
        )


def _state_dict(
    expert: TrainedExpert, in_dim: int, hidden: int, seed: int,
    quantize_head: bool, act_bits: int = _ACT_BITS,
) -> dict:
    """Build an ``UltraQuantNet.state_dict()`` from trained parameters."""
    return {
        "config": {
            "input_dim": in_dim,
            "hidden_dims": [hidden],
            "num_classes": len(expert.labels),
            "seed": seed,
            "quantize_head": quantize_head,
            "act_bits": act_bits,
        },
        "hidden": [{
            "in_dim": in_dim,
            "out_dim": hidden,
            "quantized": True,
            "w": [list(r) for r in expert.w1],
            "b": list(expert.b1),
        }],
        "head": {
            "in_dim": hidden,
            "out_dim": len(expert.labels),
            "quantized": quantize_head,
            "w": [list(r) for r in expert.w2],
            "b": list(expert.b2),
        },
    }


class ModelForge:
    """Build a model from scratch into a vault, a library and a snapshot."""

    def __init__(
        self,
        root: str | Path,
        seed: int = 0,
        hidden: int = 32,
        tier: str = "auto",
        budget_bytes: int = 1_048_576,
        quantize_head: bool = False,
        storage_uri: str | None = None,
        cache: str | int | None = None,
        n_threads: int = 0,
        part_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        """Prepare a forge rooted at ``root``.

        Args:
            root: Directory for the vault, library and archive.
            seed: Base seed; expert *i* derives ``seed + i``.
            hidden: Hidden-layer width for every expert.
            tier: ``"auto"``, ``"cuda"``, ``"cpu"`` or ``"python"``.
            budget_bytes: Resident-shard budget for the cache handed back.
            quantize_head: Ternarize the output layer as well.
            storage_uri: Where the forged library should live — an NVMe-oF
                namespace or Ceph pool rather than local files. A model built
                straight onto its serving storage never has to be copied there.
            cache: RAM tier budget in front of that storage.
            n_threads: CPU worker threads for training (0 = all cores).
            part_bytes: Maximum size of each library part file. The library is
                written as bounded, independently-readable parts rather than
                one container, so it can be synced, replicated and spread
                across volumes in pieces.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.hidden = int(hidden)
        self.tier = tier
        self.quantize_head = bool(quantize_head)
        self.n_threads = int(n_threads)
        #: How the last auto build chose its tier - config, reason,
        #: and probe timings when a probe ran.
        self.last_dispatch: dict = {}
        self.part_bytes = int(part_bytes)

        self.storage = None
        if storage_uri or cache:
            from ultraquant.storage import open_storage

            self.storage = open_storage(
                storage_uri or f"local://{(self.root / 'vault').as_posix()}", cache=cache
            )

        self.vault = ShardVault(self.root / "vault", storage=self.storage)
        self.cache = ShardCache(max_bytes=budget_bytes)
        self.router = CategoryRouter(self.vault, path=self.root / "vault" / "router.json")
        self.experts = ExpertPool(self.vault, self.cache, hidden=(self.hidden,), seed=self.seed)
        self.archive = ArTchive(self.root / "artchive")

    # ------------------------------------------------------------------ build


    # ---------------------------------------------------------------- dispatch

    def _decide_tier(self, specs: list[ExpertSpec], epochs: int,
                     batch_size: int, lr: float) -> tuple[str, int]:
        """Let the learned dispatcher pick tier and threads for this build.

        ``tier="auto"`` used to mean a fixed preference walk (CUDA, then C++,
        then Python) — the walk §6 measured losing 13x on a mixed grid. It now
        means: ask the machine's own experience. A cold start probes by
        training just the *first* expert on every available configuration —
        cheap, and a real measurement of this exact workload shape — then the
        full build runs on the winner. Every tier produces identical experts,
        so the probe wastes only the duplicated first-expert time.

        Any failure in the scheduler falls back to the static walk: dispatch
        may never break a build.

        Returns:
            ``(tier, n_threads)`` for :func:`train_experts`, with the decision
            recorded on ``self.last_dispatch``.
        """
        from ultraquant.forge import trainer as _trainer
        from ultraquant.native.scheduler import LearnedDispatch

        cpu_count = os.cpu_count() or 4
        mapping = {
            "python": ("python", 0),
            "cpp@1": ("cpu", 1),
            "cpp@half": ("cpu", max(1, cpu_count // 2)),
            "cpp@max": ("cpu", 0),
            "cuda": ("cuda", 0),
        }
        configs = ["python"]
        if _trainer._load_cpu() is not None:
            configs += ["cpp@1", "cpp@half", "cpp@max"]
        if _trainer._load_gpu() is not None:
            configs += ["cuda"]
        if len(configs) == 1:
            self.last_dispatch = {"config": "python", "reason": "only-tier"}
            return "python", 0

        dims = {
            "experts": len(specs),
            "samples": sum(len(s.xs) for s in specs) / len(specs),
            "features": len(specs[0].xs[0]),
            "hidden": self.hidden,
            "classes": sum(len(s.labels) for s in specs) / len(specs),
            "epochs": epochs,
        }
        dispatch = LearnedDispatch(
            Path(self.root) / "dispatch.json",
            available={"forge": configs}, seed=self.seed,
        )
        config, reason = dispatch.decide("forge", dims)
        if reason != "learned":
            probe_spec = [specs[0]]

            def runner(name: str):
                tier, threads = mapping[name]

                def run() -> None:
                    train_experts(
                        probe_spec, hidden=self.hidden, epochs=epochs, lr=lr,
                        batch_size=batch_size, seed=self.seed,
                        quantize_head=self.quantize_head, tier=tier,
                        n_threads=threads,
                    )

                return run

            config, timings = dispatch.probe(
                "forge", dims, {name: runner(name) for name in configs}
            )
            self.last_dispatch = {
                "config": config, "reason": "probe",
                "timings_ms": {name: round(seconds * 1e3, 2)
                               for name, seconds in sorted(timings.items())},
            }
        else:
            self.last_dispatch = {"config": config, "reason": "learned"}
        return mapping[config]

    def build(
        self,
        corpora: list[CategoryCorpus],
        epochs: int = 40,
        lr: float = 0.05,
        batch_size: int = 16,
        pack: bool = True,
        snapshot: bool = True,
    ) -> ForgeReport:
        """Forge a model from the given corpora.

        Args:
            corpora: One training corpus per category.
            epochs: Training passes per expert.
            lr: Learning rate.
            batch_size: Minibatch size.
            pack: Pack the finished shards into a ``.uql`` library.
            snapshot: Commit the build to the Ar(T)chive.

        Returns:
            A :class:`ForgeReport`.

        Raises:
            ValueError: If no corpora are supplied.
        """
        if not corpora:
            raise ValueError("nothing to forge: no corpora supplied")

        started = time.perf_counter()
        in_dim = len(corpora[0].xs[0])

        specs = [
            ExpertSpec(category=c.name, labels=list(c.labels), xs=c.xs, ys=c.ys)
            for c in corpora
        ]
        tier, n_threads = (self.tier, self.n_threads)
        if self.tier == "auto":
            try:
                tier, n_threads = self._decide_tier(specs, epochs, batch_size, lr)
            except Exception:  # noqa: BLE001 - dispatch may never break a build
                self.last_dispatch = {"config": "static-walk",
                                      "reason": "scheduler-error"}
        report = train_experts(
            specs,
            hidden=self.hidden,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=self.seed,
            quantize_head=self.quantize_head,
            tier=tier,
            n_threads=n_threads,
        )

        # -- write one shard per expert, in the format ExpertPool pages in
        parameters = 0
        per_category: dict[str, float] = {}
        for index, (corpus, trained) in enumerate(zip(corpora, report.experts)):
            payload = {
                "labels": list(trained.labels),
                "net": _state_dict(trained, in_dim, self.hidden,
                                   self.seed + index, self.quantize_head),
                "trained_examples": trained.samples,
                "prototypes": corpus.prototypes,
            }
            self.vault.add_shard(
                ExpertPool.shard_id(corpus.name), corpus.name, payload,
                kind="expert-net",
                associations={word: 0.5 for word in corpus.keywords},
            )
            self.router.register(corpus.name, list(corpus.keywords))
            # A fingerprint of what this category looks like, so a pattern can
            # find it without any words and without paging the shard.
            if corpus.prototypes:
                from ultraquant.pattern.recognition import render

                self.vault.set_signature(
                    ExpertPool.shard_id(corpus.name),
                    [render(list(rows)) for rows in corpus.prototypes.values()],
                )
            parameters += self.hidden * in_dim + self.hidden
            parameters += len(trained.labels) * self.hidden + len(trained.labels)
            per_category[corpus.name] = trained.accuracy

        self.router.save()

        # -- pack into one library and snapshot the build
        library = None
        if pack:
            lib_dir = self.root / "vault" / "library"
            result = self.vault.pack_sharded(
                lib_dir, max_part_bytes=self.part_bytes, prune_loose=True,
            )
            library = result["manifest"]

        version = None
        if snapshot:
            version = self.archive.commit("forge", {
                "tier": report.tier,
                "categories": [c.name for c in corpora],
                "experts": len(report.experts),
                "parameters": parameters,
                "hidden": self.hidden,
                "input_dim": in_dim,
                "epochs": epochs,
                "accuracy": per_category,
                "vault": self.vault.stats(),
            })

        return ForgeReport(
            tier=report.tier,
            categories=len(corpora),
            experts=len(report.experts),
            parameters=parameters,
            train_seconds=report.seconds,
            total_seconds=time.perf_counter() - started,
            mean_accuracy=report.mean_accuracy,
            library=library,
            snapshot=version,
            store_bytes=self.vault.stats()["total_bytes"],
            per_category=per_category,
        )

    # ----------------------------------------------------------------- expand

    def expand(
        self,
        corpora: list[CategoryCorpus],
        epochs: int = 40,
        lr: float = 0.05,
        batch_size: int = 16,
        retrain: bool = False,
        snapshot: bool = True,
    ) -> ForgeReport:
        """Add categories to a library that already exists.

        A model is not finished when it is first built — new categories arrive,
        and rebuilding the whole library to accommodate them would be absurd once
        it is hundreds of gigabytes. So expansion trains only what is new and
        writes it into a **fresh library segment**, leaving existing segments
        untouched.

        Immutable segments are what make this safe: readers holding offsets into
        an older segment stay valid, nothing is rewritten in place, and a failed
        expansion cannot corrupt what was already there. The cost is that
        segments accumulate, which is what
        :meth:`~ultraquant.shards.vault.ShardVault.compact` is for.

        Args:
            corpora: Corpora for the categories to add.
            epochs: Training passes per new expert.
            lr: Learning rate.
            batch_size: Minibatch size.
            retrain: Also rebuild categories that already exist. Their new shard
                shadows the old one, whose bytes are reclaimed at compaction.
            snapshot: Commit the expansion to the Ar(T)chive.

        Returns:
            A :class:`ForgeReport` covering only the categories actually trained.

        Raises:
            ValueError: If no corpora are supplied.
        """
        if not corpora:
            raise ValueError("nothing to expand: no corpora supplied")

        pending = [
            corpus for corpus in corpora
            if retrain or not self.vault.has(ExpertPool.shard_id(corpus.name))
        ]
        if not pending:
            return ForgeReport(
                tier="none", categories=0, experts=0, parameters=0,
                train_seconds=0.0, total_seconds=0.0, mean_accuracy=0.0,
                store_bytes=self.vault.stats()["total_bytes"],
            )

        started = time.perf_counter()
        existing = len(self.vault.catalog())
        report = self._train_and_store(pending, epochs, lr, batch_size)

        # New shards go into their own segment; earlier ones are never rewritten.
        lib_dir = self.root / "vault" / "library"
        lib_dir.mkdir(parents=True, exist_ok=True)
        wave = len({p.name.split("-")[0] for p in lib_dir.glob("*.uql")}) + 1
        fresh = [ExpertPool.shard_id(c.name) for c in pending]
        result = self.vault.pack_sharded(
            lib_dir, fresh, max_part_bytes=self.part_bytes,
            prune_loose=True, prefix=f"seg{wave:03d}",
        )
        target = result["manifest"]
        self.router.save()

        version = None
        if snapshot:
            version = self.archive.commit("expand", {
                "added": [c.name for c in pending],
                "parts": result["part_files"],
                "categories_before": existing,
                "categories_after": len(self.vault.catalog()),
                "accuracy": report["per_category"],
                "vault": self.vault.stats(),
            })

        return ForgeReport(
            tier=report["tier"],
            categories=len(pending),
            experts=len(pending),
            parameters=report["parameters"],
            train_seconds=report["train_seconds"],
            total_seconds=time.perf_counter() - started,
            mean_accuracy=report["mean_accuracy"],
            library=str(target),
            snapshot=version,
            store_bytes=self.vault.stats()["total_bytes"],
            per_category=report["per_category"],
        )

    def compact(self) -> dict:
        """Merge every library segment into one and reclaim stranded bytes."""
        lib_dir = self.root / "vault" / "library"
        lib_dir.mkdir(parents=True, exist_ok=True)
        return self.vault.compact(lib_dir / "compacted.uql")

    def library_parts(self) -> list[str]:
        """The part files this library is currently spread across."""
        return self.vault.libraries()

    def _train_and_store(
        self, corpora: list[CategoryCorpus], epochs: int, lr: float, batch_size: int
    ) -> dict:
        """Train the given corpora and write their shards. Shared by build/expand."""
        in_dim = len(corpora[0].xs[0])
        specs = [
            ExpertSpec(category=c.name, labels=list(c.labels), xs=c.xs, ys=c.ys)
            for c in corpora
        ]
        tier, n_threads = (self.tier, self.n_threads)
        if self.tier == "auto":
            try:
                tier, n_threads = self._decide_tier(specs, epochs, batch_size, lr)
            except Exception:  # noqa: BLE001 - dispatch may never break a build
                self.last_dispatch = {"config": "static-walk",
                                      "reason": "scheduler-error"}
        result = train_experts(
            specs, hidden=self.hidden, epochs=epochs, lr=lr, batch_size=batch_size,
            seed=self.seed, quantize_head=self.quantize_head, tier=tier,
            n_threads=n_threads,
        )

        parameters = 0
        per_category: dict[str, float] = {}
        base = len(self.vault.catalog())
        for offset, (corpus, trained) in enumerate(zip(corpora, result.experts)):
            payload = {
                "labels": list(trained.labels),
                "net": _state_dict(trained, in_dim, self.hidden,
                                   self.seed + base + offset, self.quantize_head),
                "trained_examples": trained.samples,
                "prototypes": corpus.prototypes,
            }
            self.vault.add_shard(
                ExpertPool.shard_id(corpus.name), corpus.name, payload,
                kind="expert-net",
                associations={word: 0.5 for word in corpus.keywords},
            )
            self.router.register(corpus.name, list(corpus.keywords))
            # A fingerprint of what this category looks like, so a pattern can
            # find it without any words and without paging the shard.
            if corpus.prototypes:
                from ultraquant.pattern.recognition import render

                self.vault.set_signature(
                    ExpertPool.shard_id(corpus.name),
                    [render(list(rows)) for rows in corpus.prototypes.values()],
                )
            parameters += self.hidden * in_dim + self.hidden
            parameters += len(trained.labels) * self.hidden + len(trained.labels)
            per_category[corpus.name] = trained.accuracy

        return {
            "tier": result.tier,
            "parameters": parameters,
            "train_seconds": result.seconds,
            "mean_accuracy": result.mean_accuracy,
            "per_category": per_category,
        }

    # ----------------------------------------------------------------- verify

    def verify(self, corpora: list[CategoryCorpus], samples: int = 40) -> dict[str, float]:
        """Re-check the forged model by paging experts back in from the store.

        This exercises the real serving path — catalog lookup, chunked read,
        digest check, cache admission — rather than the in-memory weights, so it
        proves the shards on disk actually work.

        Args:
            corpora: The corpora the model was built from.
            samples: Samples per category to check.

        Returns:
            ``{category: accuracy}`` measured through the vault.
        """
        out: dict[str, float] = {}
        for corpus in corpora:
            correct = 0
            total = min(samples, len(corpus.xs))
            for x, y in zip(corpus.xs[:total], corpus.ys[:total]):
                label, _confidence = self.experts.predict(corpus.name, x)
                if label == corpus.labels[y]:
                    correct += 1
            out[corpus.name] = correct / max(total, 1)
        return out
