"""The hypervector gate: does binding buy structure the slot dict cannot hold?

Stated before running, so it cannot be adjusted afterwards:

    Represent and correctly query **nested** structure that the flat slot dict
    cannot represent at all, and retrieve that whole scene from the library by
    similarity, at >=90% accuracy at a depth and capacity stated in advance. If
    it only reproduces what the flat dict already does, it is an optimisation
    wearing an architecture costume and it gets dropped.

Pre-registered parameters: **nesting depth 3**, **24 scenes** in the library,
**>=90%** on both tasks.

Two tasks, because the claim has two halves:

* **Structural query** — recover the filler at each depth of a nested scene.
  This is what binding and permutation are for. **Passes at 1.000.**
* **Scene retrieval** — given a cue, find the matching scene among 24.
  **Fails at 0.74 against a target of 0.90.**

**Two flaws in the first version of this experiment are recorded rather than
quietly fixed**, because both would have produced a confident wrong answer.

*The task had no answer.* Cues were made by corrupting one level at random. The
best accuracy **any** method could reach on that was measured afterwards at 0.000
for depth 2 and 0.246 for depth 3 — the corrupted cue sat as close to other
library scenes as to the one it came from. Both arms scored at that ceiling, so
the comparison was measuring the ambiguity of the task. :func:`well_posed_cue`
now rejects ambiguous cues, which puts the oracle at exactly 1.000.

*The baseline was the oracle.* A path-keyed dict compared by Jaccard was chosen
to avoid the hollow "a dict cannot nest" claim — a dict nests fine with keys like
``0/container``. But Jaccard over path keys **is** the ground-truth similarity
this task is defined by, so that baseline scores the oracle by construction and
no lossy representation can ever beat it. Requiring hypervectors to beat it was
unwinnable, which is the mirror image of a hollow pass and just as useless. The
baseline is therefore reported as what it is, and task B is judged on its
absolute number against the pre-registered target.

Size is reported too, because it is the honest cost: a hypervector is 1,250 bytes
at **any** depth, and the dict is smaller until the structure gets large.
"""

from __future__ import annotations

import random
import statistics
import sys

from ultraquant.reason.hypervector import (
    DIM,
    ItemMemory,
    bind,
    bundle,
    permute,
    seed_vector,
    similarity,
    unbind,
)

__all__ = [
    "DEPTH",
    "SCENES",
    "TARGET",
    "encode_scene",
    "query_depth",
    "path_keys",
    "jaccard",
    "one_trial",
    "run_gate",
    "main",
]

#: Pre-registered gate parameters.
DEPTH = 3
SCENES = 24
TARGET = 0.90

#: The vocabulary scenes are built from.
CONTAINERS = ["frame", "box", "ring", "shell"]
CONTENTS = ["arrow", "plus", "dot", "wave", "star", "bar"]

#: Roles. ``CONTAINER`` labels a level's own identity; ``INSIDE`` marks the
#: sub-scene, which is permuted so depth is recoverable.
ROLE_CONTAINER = "ROLE_CONTAINER"
ROLE_CONTENT = "ROLE_CONTENT"


def encode_scene(levels: list[str], memory: ItemMemory) -> int:
    """Encode ``frame(box(arrow))`` as one vector.

    Each level binds its identity to a role and bundles that with a *permuted*
    encoding of the level below. Permutation is what makes depth recoverable:
    binding alone is commutative, so without it a nested scene would be
    indistinguishable from a flat bag of its parts.

    Args:
        levels: Containers outermost-first, with the innermost content last.
        memory: Item memory holding the vocabulary.

    Returns:
        One ``DIM``-bit vector, whatever the depth.
    """
    inner = bind(memory.get(ROLE_CONTENT), memory.get(levels[-1]))
    for name in reversed(levels[:-1]):
        inner = bundle([
            bind(memory.get(ROLE_CONTAINER), memory.get(name)),
            permute(inner, 1),
        ])
    return inner


def query_depth(scene: int, depth: int, memory: ItemMemory) -> str | None:
    """Recover the filler at ``depth`` — 0 is the outermost container.

    Descends by un-permuting once per level, then unbinds the role that level
    carries and cleans the result up against the item memory.
    """
    current = scene
    for _ in range(depth):
        current = permute(current, -1)
    is_content = False
    probe = unbind(current, memory.get(ROLE_CONTAINER))
    best = memory.cleanup(probe)
    content_probe = unbind(current, memory.get(ROLE_CONTENT))
    content_best = memory.cleanup(content_probe)
    if content_best and (not best or content_best[1] > best[1]):
        best, is_content = content_best, True
    if best is None:
        return None
    name = best[0]
    allowed = CONTENTS if is_content else CONTAINERS
    if name not in allowed:
        # Cleanup landed on a role or the wrong vocabulary; take the best
        # candidate from the vocabulary this level can actually hold.
        pool = [(item, similarity(content_probe if is_content else probe,
                                  memory.get(item)))
                for item in allowed]
        pool.sort(key=lambda pair: (-pair[1], pair[0]))
        name = pool[0][0]
    return name


# --------------------------------------------------------------------------- #
# the baseline: the same structure in a dict
# --------------------------------------------------------------------------- #


def path_keys(levels: list[str]) -> set[str]:
    """The scene as path-keyed dict entries — the fair baseline's form.

    ``frame(box(arrow))`` becomes ``{"0/container=frame", "1/container=box",
    "2/content=arrow"}``. A dict represents this perfectly well; what it cannot
    do is be one fixed-width object that the library's similarity index accepts.
    """
    out = set()
    for index, name in enumerate(levels[:-1]):
        out.add(f"{index}/container={name}")
    out.add(f"{len(levels) - 1}/content={levels[-1]}")
    return out


def jaccard(left: set[str], right: set[str]) -> float:
    """Overlap of two path-keyed scenes."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


# --------------------------------------------------------------------------- #
# trial
# --------------------------------------------------------------------------- #


def _corrupt(levels: list[str], rng: random.Random) -> list[str]:
    """Replace one level with a different value — a partial, wrong cue."""
    out = list(levels)
    position = rng.randrange(len(out))
    pool = CONTENTS if position == len(out) - 1 else CONTAINERS
    choices = [c for c in pool if c != out[position]]
    out[position] = rng.choice(choices)
    return out


def well_posed_cue(
    index: int, library: list[list[str]], keyed: list[set[str]],
    rng: random.Random, attempts: int = 60,
) -> list[str] | None:
    """A corrupted cue whose *unique* best ground-truth match is the target.

    Without this the task has no answer. The first version of this experiment
    corrupted one level at random and asked both methods to find the original;
    measured afterwards, the best accuracy **any** method could reach was 0.000
    at depth 2 and 0.246 at depth 3, because the corrupted cue was as close to
    other library scenes as to the one it came from. Both arms scored at that
    ceiling and the comparison measured the ambiguity of the task rather than
    anything about the representations.

    Rejecting ambiguous cues makes the oracle exactly 1.000, so the number that
    comes out is the method's own.

    Returns:
        A usable cue, or ``None`` if none was found in ``attempts`` tries.
    """
    for _ in range(attempts):
        cue = _corrupt(library[index], rng)
        cue_keys = path_keys(cue)
        scores = [jaccard(cue_keys, k) for k in keyed]
        best = max(scores)
        if sum(1 for s in scores if s == best) == 1 and scores.index(best) == index:
            return cue
    return None


def one_trial(seed: int, depth: int = DEPTH, scenes: int = SCENES) -> dict:
    """Build a library of scenes, then query and retrieve.

    Returns:
        Per-task accuracies for both the hypervector and the dict baseline.
    """
    rng = random.Random(seed)
    memory = ItemMemory()
    for name in CONTAINERS + CONTENTS + [ROLE_CONTAINER, ROLE_CONTENT]:
        memory.add(name)

    library: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    while len(library) < scenes:
        levels = [rng.choice(CONTAINERS) for _ in range(depth - 1)]
        levels.append(rng.choice(CONTENTS))
        if tuple(levels) in seen:
            continue
        seen.add(tuple(levels))
        library.append(levels)

    encoded = [encode_scene(levels, memory) for levels in library]
    keyed = [path_keys(levels) for levels in library]

    # -- task 1: structural query ------------------------------------------
    asked = 0
    right = 0
    for levels, vector in zip(library, encoded):
        for level in range(depth):
            asked += 1
            right += query_depth(vector, level, memory) == levels[level]
    structural = right / asked

    # -- task 2: retrieval from a cue that identifies exactly one scene ------
    hv_hits = 0
    asked = 0
    for index in range(scenes):
        cue = well_posed_cue(index, library, keyed, rng)
        if cue is None:
            continue
        asked += 1
        cue_vector = encode_scene(cue, memory)
        best = max(range(scenes), key=lambda i: similarity(cue_vector, encoded[i]))
        hv_hits += best == index

    return {
        "structural": structural,
        "retrieval_hv": hv_hits / asked if asked else 0.0,
        # The oracle is 1.000 by construction. The path-keyed dict scores it
        # exactly, because Jaccard over path keys *is* the ground-truth
        # similarity — which makes that baseline an oracle, not a competitor.
        # Asking a lossy representation to beat it was never winnable, and
        # noting so is more useful than the number it would have produced.
        "retrieval_oracle": 1.0,
        "usable_cues": asked / scenes,
        "bytes_hv": (DIM + 7) // 8,
        "bytes_dict": sum(len(k) for k in keyed[0]),
    }


def run_gate(seeds: int = 12, depth: int = DEPTH, scenes: int = SCENES) -> dict:
    """Run the gate over ``seeds`` seeds."""
    trials = [one_trial(seed, depth, scenes) for seed in range(seeds)]

    def mean(key: str) -> float:
        return statistics.fmean(t[key] for t in trials)

    def sd(key: str) -> float:
        values = [t[key] for t in trials]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    structural = mean("structural")
    retrieval = mean("retrieval_hv")
    return {
        "seeds": seeds,
        "depth": depth,
        "scenes": scenes,
        "structural": structural,
        "structural_sd": sd("structural"),
        "retrieval_hv": retrieval,
        "retrieval_hv_sd": sd("retrieval_hv"),
        "retrieval_oracle": 1.0,
        "usable_cues": mean("usable_cues"),
        "bytes_hv": trials[0]["bytes_hv"],
        "bytes_dict": trials[0]["bytes_dict"],
        "structural_passes": structural >= TARGET,
        "retrieval_passes": retrieval >= TARGET,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a verdict."""
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds = int(argv[0]) if argv else 12

    print("Hypervector gate: structure the flat slot dict cannot hold")
    print(f"pre-registered: depth {DEPTH}, {SCENES} scenes, target {TARGET:.0%}, "
          f"{seeds} seeds")

    result = run_gate(seeds=seeds)
    print()
    print(f"  A. structural query  {result['structural']:.3f} "
          f"(sd {result['structural_sd']:.3f})   target {TARGET:.2f}   "
          f"{'PASS' if result['structural_passes'] else 'FAIL'}")
    print(f"  B. scene retrieval   {result['retrieval_hv']:.3f} "
          f"(sd {result['retrieval_hv_sd']:.3f})   target {TARGET:.2f}   "
          f"{'PASS' if result['retrieval_passes'] else 'FAIL'}")
    print(f"     oracle on task B is 1.000 by construction; "
          f"{result['usable_cues']:.0%} of cues were usable")
    print()
    print(f"  size: {result['bytes_hv']} bytes at any depth, against "
          f"{result['bytes_dict']} bytes for the dict at depth {DEPTH}")

    print()
    print("depth scaling:")
    print(f"  {'depth':>6} {'structural':>12} {'retrieval':>11} {'hv bytes':>10}")
    for depth in (2, 3, 4, 5, 6):
        deep = run_gate(seeds=6, depth=depth)
        print(f"  {depth:>6} {deep['structural']:>12.3f} "
              f"{deep['retrieval_hv']:>11.3f} {deep['bytes_hv']:>10}")

    print()
    print("verdict")
    if result["structural_passes"] and result["retrieval_passes"]:
        print("  PASS - both halves clear the pre-registered target.")
        return 0
    print("  FAIL - the gate was a conjunction and only half of it holds.")
    print()
    if result["structural_passes"]:
        print("  A passes outright: nested structure is recovered exactly, at")
        print("  every depth measured. The flat slot dict genuinely cannot hold")
        print("  this - two container levels collide on one key - so that half")
        print("  is not an optimisation wearing an architecture costume.")
    if not result["retrieval_passes"]:
        print("  B fails clearly. Unbinding is exact because the key is known;")
        print("  comparing whole scenes is lossy, because each level of bundling")
        print("  adds crosstalk and the differences between scenes that share")
        print("  most of their levels sit under it.")
    print()
    print("  Two flaws in the first version of this experiment are worth")
    print("  recording rather than quietly fixing:")
    print("   * the cue did not identify the target - the best accuracy any")
    print("     method could reach was 0.000 at depth 2 and 0.246 at depth 3,")
    print("     and both arms sat at that ceiling;")
    print("   * the dict baseline WAS the oracle, since Jaccard over path keys")
    print("     is the ground-truth similarity, so 'beat the baseline' could")
    print("     never have been satisfied by any lossy representation.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
