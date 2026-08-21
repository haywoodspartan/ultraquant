"""ExpertPool -- per-category ternary experts stored as vault shards.

Realises the mixture-of-experts / parameter-paging idea from
SPEC-INTERPRETER Section 4: each category owns a single
:class:`~ultraquant.model.network.UltraQuantNet`, persisted as one shard in a
:class:`~ultraquant.shards.vault.ShardVault` under the id ``expert:{category}``.
Experts are loaded on demand through a
:class:`~ultraquant.shards.budget.ShardCache`, so that of the ``N`` experts on
disk only a handful are ever resident in RAM at once -- the same "load a chunk,
swap in and out on the fly" story the shard library tells at scale.

Budget currency: the loader reports each expert's *stored* ``nbytes`` (the
compressed size recorded in the vault catalog), so the ShardCache accounts for
resident experts in exactly the units the vault persists them in.

Pure Python standard library only.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from ultraquant.model.network import UltraQuantNet

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from ultraquant.shards.budget import ShardCache
    from ultraquant.shards.vault import ShardVault

__all__ = ["ExpertPool"]

# Largest positive value we seed an :class:`UltraQuantNet` with; keeps derived
# per-category seeds inside a comfortable 31-bit range.
_INT32_MAX = 2 ** 31 - 1

# Minibatch size used while training an expert (mirrors SPEC.md Section 10).
_MINIBATCH = 16


class ExpertPool:
    """A pool of per-category experts backed by vault shards.

    Each category is represented by one :class:`UltraQuantNet` serialised into a
    shard whose payload is ``{"labels": [...], "net": <state_dict>,
    "trained_examples": int}``.  Prediction routes through a
    :class:`ShardCache`, so experts are paged in on a cache miss and served from
    RAM on a hit; training rewrites the shard and invalidates the cache entry.
    """

    def __init__(
        self,
        vault: "ShardVault",
        cache: "ShardCache",
        input_dim: int = 30,
        hidden: tuple[int, ...] = (32,),
        seed: int = 0,
    ) -> None:
        """Build a pool over an existing vault and cache.

        Args:
            vault: Shard store that persists each expert (source of truth).
            cache: Resident-set manager experts are loaded through on demand.
            input_dim: Feature dimensionality of every freshly created expert.
            hidden: Hidden-layer sizes for every freshly created expert.
            seed: Base seed; each category derives a distinct, deterministic
                weight-initialisation seed from it.
        """
        self.vault = vault
        self.cache = cache
        self.input_dim = int(input_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.seed = int(seed)

    # -- ids -------------------------------------------------------------
    @staticmethod
    def shard_id(category: str) -> str:
        """Return the vault shard id for a category (``expert:{category}``)."""
        return f"expert:{category}"

    # -- introspection ---------------------------------------------------
    def has_expert(self, category: str) -> bool:
        """Return True if a shard for ``category`` already exists in the vault."""
        return self.vault.has(self.shard_id(category))

    def _stored_nbytes(self, shard_id: str) -> int:
        """Return the shard's stored (compressed) size from the vault catalog.

        This is the budget currency the :class:`ShardCache` accounts in.
        """
        return int(self.vault.entry(shard_id)["nbytes"])

    # -- net (de)serialisation ------------------------------------------
    def _net_seed(self, category: str) -> int:
        """Deterministically derive a per-category weight-init seed."""
        return random.Random(f"{self.seed}|expert|{category}").randrange(1, _INT32_MAX)

    def _new_net(
        self, category: str, num_classes: int, input_dim: int | None = None
    ) -> UltraQuantNet:
        """Create a fresh, seeded expert net for ``category``.

        Args:
            category: Category the expert specialises in.
            num_classes: Size of the label set.
            input_dim: Feature width for *this* expert. ``None`` uses the pool
                default, which is what a single-modality library wants.
        """
        return UltraQuantNet(
            input_dim=int(input_dim) if input_dim else self.input_dim,
            hidden_dims=list(self.hidden),
            num_classes=num_classes,
            seed=self._net_seed(category),
        )

    @staticmethod
    def _rebuild_net(net_state: dict) -> UltraQuantNet:
        """Rebuild a net from a stored ``state_dict`` (architecture from config)."""
        cfg = net_state["config"]
        net = UltraQuantNet(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=[int(h) for h in cfg["hidden_dims"]],
            num_classes=int(cfg["num_classes"]),
            seed=int(cfg.get("seed", 0)),
            quantize_head=bool(cfg.get("quantize_head", False)),
            act_bits=int(cfg.get("act_bits", 8)),
        )
        net.load_state_dict(net_state)
        return net

    # -- public API ------------------------------------------------------
    def ensure_expert(
        self, category: str, labels: list[str], input_dim: int | None = None
    ) -> dict:
        """Create and persist a fresh expert for ``category`` if absent.

        Args:
            category: Category the expert specialises in.
            labels: Ordered class labels; index ``i`` is class ``i``.
            input_dim: Feature width for this expert, when it differs from the
                pool default — see :meth:`train_expert`.

        Returns:
            The vault catalog entry for the expert (newly created or existing).
        """
        sid = self.shard_id(category)
        if self.vault.has(sid):
            return self.vault.entry(sid)
        net = self._new_net(category, len(labels), input_dim)
        payload = {
            "labels": list(labels),
            "net": net.state_dict(),
            "trained_examples": 0,
        }
        return self.vault.add_shard(sid, category, payload, kind="expert-net")

    def _loader(self, shard_id: str) -> Callable[[], tuple[dict, int]]:
        """Build the ShardCache loader for a shard (payload + stored nbytes)."""

        def load() -> tuple[dict, int]:
            payload = self.vault.get(shard_id)  # verifies sha256; bumps touch()
            nbytes = self._stored_nbytes(shard_id)
            return payload, nbytes

        return load

    def predict(self, category: str, features: list[float]) -> tuple[str, float]:
        """Predict a label for ``features`` using ``category``'s expert.

        The expert is fetched through the cache (paged in on a miss, served from
        RAM on a hit), its net rebuilt from the cached state, and the argmax
        label returned with its softmax confidence.

        Args:
            category: Category whose expert should classify the features.
            features: Feature vector of length ``input_dim``.

        Returns:
            ``(label_name, confidence)``.

        Raises:
            KeyError: If no expert has been created for ``category`` yet.
        """
        sid = self.shard_id(category)
        if not self.has_expert(category):
            raise KeyError(
                f"no expert for category {category!r}; call ensure_expert first"
            )
        payload = self.cache.get(sid, self._loader(sid))
        net = self._rebuild_net(payload["net"])
        labels = payload["labels"]
        expected = int(payload["net"]["config"]["input_dim"])
        if len(features) != expected:
            # Once experts may have different feature widths, a mismatch is a
            # routing error rather than a caller typo, and it must not be
            # allowed to look like a confident answer.
            raise ValueError(
                f"expert {category!r} takes {expected} features, got "
                f"{len(features)}; this pattern belongs to a different modality"
            )
        vram = getattr(self, "vram", None)
        if vram is not None:
            # The hot tier (§11.45): resident layers answer with only
            # activations crossing the bus, bit-exact against the Python
            # path, and any refusal falls straight through to it.
            got = net.predict_vram(list(features), vram, sid)
            if got is not None:
                idx, conf = got
                return labels[idx], conf
        idx, conf = net.predict(list(features))
        return labels[idx], conf

    def _invalidate_vram(self, sid: str) -> None:
        """Take a rewritten expert's resident layers down with it."""
        vram = getattr(self, "vram", None)
        if vram is not None:
            vram.drop_prefix(sid)

    def train_expert(
        self,
        category: str,
        xs: list[list[float]],
        ys: list[int],
        labels: list[str],
        epochs: int = 30,
        lr: float = 0.05,
    ) -> dict:
        """Train ``category``'s expert, persist it, and invalidate its cache slot.

        Training starts from the stored weights when the expert already exists
        (continued learning) or from a fresh seeded net otherwise.  The updated
        expert is written back to the vault -- which preserves the shard's
        access_count/associations across the rewrite -- and its cache entry is
        invalidated so the next :meth:`predict` pages in the new weights.

        Args:
            category: Category to train.
            xs: Training feature vectors.
            ys: Integer class labels aligned with ``xs`` (each in
                ``range(len(labels))``).
            labels: Ordered class labels stored alongside the net.
            epochs: Number of passes over the data.
            lr: SGD learning rate.

        Returns:
            ``{"loss": <final-epoch mean loss>, "accuracy": <train accuracy>}``.
        """
        sid = self.shard_id(category)
        if self.has_expert(category):
            stored = self.vault.get(sid)
            net = self._rebuild_net(stored["net"])
            prior_examples = int(stored.get("trained_examples", 0))
        else:
            # The data decides the feature width, not the pool. A pool-wide
            # input_dim means every expert in a library must perceive the same
            # kind of thing, which is exactly the limit worth removing: one
            # catalogued library should be able to hold a 30-input glyph expert
            # beside a 512-input text expert without either knowing about the
            # other. Stored nets already carry their own config, so paging,
            # routing and training all work unchanged.
            net = self._new_net(category, len(labels), len(xs[0]) if xs else None)
            prior_examples = 0

        rng = random.Random(f"{self.seed}|train|{category}")
        xs = [list(x) for x in xs]
        ys = list(ys)
        loss = self._train_net(net, xs, ys, epochs, lr, rng)
        accuracy = self._accuracy(net, xs, ys)

        payload = {
            "labels": list(labels),
            "net": net.state_dict(),
            "trained_examples": prior_examples + len(xs),
        }
        self.vault.add_shard(sid, category, payload, kind="expert-net")
        self.cache.invalidate(sid)
        self._invalidate_vram(sid)
        return {"loss": loss, "accuracy": accuracy}

    # -- training helpers ------------------------------------------------
    @staticmethod
    def _train_net(
        net: UltraQuantNet,
        xs: list[list[float]],
        ys: list[int],
        epochs: int,
        lr: float,
        rng: random.Random,
    ) -> float:
        """Train ``net`` in shuffled minibatches; return final-epoch mean loss."""
        n = len(xs)
        if n == 0:
            return 0.0
        final_loss = 0.0
        for _ in range(max(epochs, 0)):
            order = list(range(n))
            rng.shuffle(order)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, _MINIBATCH):
                chunk = order[start : start + _MINIBATCH]
                bx = [xs[i] for i in chunk]
                by = [ys[i] for i in chunk]
                epoch_loss += net.train_batch(bx, by, lr)
                batches += 1
            final_loss = epoch_loss / batches if batches else 0.0
        return final_loss

    @staticmethod
    def _accuracy(net: UltraQuantNet, xs: list[list[float]], ys: list[int]) -> float:
        """Fraction of ``xs`` whose argmax prediction matches ``ys``."""
        if not xs:
            return 0.0
        correct = sum(1 for x, y in zip(xs, ys) if net.predict(x)[0] == y)
        return correct / len(xs)
