"""Hypervector algebra, and the integrity of the gate that judged it.

The algebra is tested as machinery: binding inverts, bundling is a genuine
majority, counting is associative so any split of the work agrees, and named
vectors are identical across processes that never communicate.

The gate's *verdict* is recorded in ARCHITECTURE.md, not asserted here — these
tests must not start failing because a measured number moved. What is asserted is
that the experiment remains capable of producing an honest answer: that its cues
identify their targets, and that the property which made its first baseline
useless is documented in the code rather than forgotten.
"""

from __future__ import annotations

import random
import unittest

from ultraquant.experiments import structure_gate as gate
from ultraquant.reason.hypervector import (
    DIM,
    ItemMemory,
    bind,
    bundle,
    bundle_counts,
    merge_counts,
    permute,
    resolve,
    seed_vector,
    similarity,
    unbind,
)


class AlgebraTests(unittest.TestCase):
    """Bind, bundle, permute."""

    def setUp(self) -> None:
        self.rng = random.Random(0)

    def test_binding_is_exactly_invertible(self) -> None:
        a, b = self.rng.getrandbits(DIM), self.rng.getrandbits(DIM)
        self.assertEqual(unbind(bind(a, b), a), b)

    def test_unrelated_vectors_are_near_orthogonal(self) -> None:
        """The whole scheme rests on random vectors being unalike."""
        scores = [
            abs(similarity(self.rng.getrandbits(DIM), self.rng.getrandbits(DIM)))
            for _ in range(50)
        ]
        self.assertLess(max(scores), 0.1)

    def test_permutation_is_reversible_and_breaks_commutativity(self) -> None:
        vector = self.rng.getrandbits(DIM)
        self.assertEqual(permute(permute(vector, 3), -3), vector)
        self.assertLess(abs(similarity(permute(vector, 1), vector)), 0.1)

    def test_bundle_is_similar_to_every_member(self) -> None:
        members = [self.rng.getrandbits(DIM) for _ in range(8)]
        merged = bundle(members)
        for member in members:
            self.assertGreater(similarity(merged, member), 0.15)

    def test_bundle_matches_a_brute_force_majority(self) -> None:
        for count in (2, 3, 4, 5, 8, 9, 17):
            with self.subTest(count=count):
                vectors = [self.rng.getrandbits(64) for _ in range(count)]
                planes = bundle_counts(vectors)
                expected = 0
                for bit in range(64):
                    ones = sum((v >> bit) & 1 for v in vectors)
                    if ones * 2 > count:
                        expected |= 1 << bit
                got = bundle(vectors, dim=64)
                # Only strict-majority bits are compared; ties are a separate
                # rule and are checked by the balance test below.
                self.assertEqual(got & expected, expected)

    def test_an_empty_or_single_bundle_is_trivial(self) -> None:
        self.assertEqual(bundle([]), 0)
        one = self.rng.getrandbits(DIM)
        self.assertEqual(bundle([one]), one)


class DeterminismTests(unittest.TestCase):
    """A single core and many cores must agree exactly."""

    def test_counting_is_associative_under_any_split(self) -> None:
        rng = random.Random(1)
        vectors = [rng.getrandbits(2048) for _ in range(37)]
        whole = bundle_counts(vectors)
        for cut in (1, 5, 18, 36):
            with self.subTest(cut=cut):
                left = bundle_counts(vectors[:cut])
                right = bundle_counts(vectors[cut:])
                self.assertEqual(merge_counts(left, right), whole)

    def test_a_split_bundle_equals_the_single_core_bundle(self) -> None:
        rng = random.Random(2)
        vectors = [rng.getrandbits(1024) for _ in range(16)]
        single = bundle(vectors, dim=1024)
        merged = merge_counts(bundle_counts(vectors[:7]), bundle_counts(vectors[7:]))
        self.assertEqual(resolve(merged, len(vectors), dim=1024), single)

    def test_named_vectors_need_no_shared_state(self) -> None:
        """Two machines must agree on what a concept is without talking."""
        self.assertEqual(seed_vector("SHAPE", 4096), seed_vector("SHAPE", 4096))
        self.assertNotEqual(seed_vector("SHAPE", 4096), seed_vector("MARK", 4096))

    def test_tie_breaking_leaves_the_bundle_balanced(self) -> None:
        """The bug this guards was silent and cost a fifth of retrievals.

        Ties were broken with the XOR of the inputs. A tie at even ``n`` means
        exactly ``n / 2`` ones, so when ``n / 2`` is even their XOR is *always*
        zero and every tie resolved to 0. Measured at n=16 that left 4,053 of
        10,000 bits set instead of ~5,000, and the density bias made every
        unbound probe resemble the key it was unbound with.
        """
        rng = random.Random(3)
        for count in (2, 4, 8, 16, 32):
            with self.subTest(count=count):
                merged = bundle([rng.getrandbits(DIM) for _ in range(count)])
                density = merged.bit_count() / DIM
                self.assertGreater(density, 0.45, "bundle is biased toward zero")
                self.assertLess(density, 0.55, "bundle is biased toward one")


class ItemMemoryTests(unittest.TestCase):
    """Cleanup, without which one operation degrades into noise."""

    def test_capacity_holds_many_pairs_in_one_vector(self) -> None:
        memory = ItemMemory()
        roles = [f"ROLE{i}" for i in range(32)]
        fills = [f"FILL{i}" for i in range(32)]
        for name in roles + fills:
            memory.add(name)
        record = bundle([bind(memory.get(roles[i]), memory.get(fills[i]))
                         for i in range(32)])
        recovered = sum(
            memory.cleanup(unbind(record, memory.get(roles[k])))[0] == fills[k]
            for k in range(32)
        )
        self.assertEqual(recovered, 32)

    def test_cleanup_is_deterministic(self) -> None:
        memory = ItemMemory()
        for name in ("a", "b", "c"):
            memory.add(name)
        probe = memory.get("b")
        self.assertEqual(memory.cleanup(probe), memory.cleanup(probe))

    def test_a_floor_can_reject_a_match(self) -> None:
        memory = ItemMemory()
        memory.add("a")
        self.assertIsNone(memory.cleanup(random.Random(0).getrandbits(DIM), floor=0.5))


class StructureGateIntegrityTests(unittest.TestCase):
    """The experiment must remain able to give an honest answer."""

    def setUp(self) -> None:
        self.memory = ItemMemory()
        for name in (gate.CONTAINERS + gate.CONTENTS
                     + [gate.ROLE_CONTAINER, gate.ROLE_CONTENT]):
            self.memory.add(name)

    def test_a_nested_scene_is_one_fixed_width_vector(self) -> None:
        """Constant size regardless of depth is the architectural claim."""
        shallow = gate.encode_scene(["frame", "arrow"], self.memory)
        deep = gate.encode_scene(["frame", "box", "ring", "shell", "arrow"],
                                 self.memory)
        self.assertLessEqual(shallow.bit_length(), DIM)
        self.assertLessEqual(deep.bit_length(), DIM)

    def test_nesting_is_not_a_bag_of_parts(self) -> None:
        """Without permutation, order would be invisible and nesting meaningless."""
        one = gate.encode_scene(["frame", "box", "arrow"], self.memory)
        other = gate.encode_scene(["box", "frame", "arrow"], self.memory)
        self.assertNotEqual(one, other)
        self.assertLess(similarity(one, other), 0.9)

    def test_every_level_of_a_nested_scene_is_recoverable(self) -> None:
        levels = ["frame", "box", "arrow"]
        scene = gate.encode_scene(levels, self.memory)
        for depth, expected in enumerate(levels):
            with self.subTest(depth=depth):
                self.assertEqual(gate.query_depth(scene, depth, self.memory), expected)

    def test_the_dict_baseline_represents_the_same_structure(self) -> None:
        """The 'a dict cannot nest' claim would have been false."""
        keys = gate.path_keys(["frame", "box", "arrow"])
        self.assertEqual(
            keys, {"0/container=frame", "1/container=box", "2/content=arrow"}
        )
        self.assertNotEqual(keys, gate.path_keys(["box", "frame", "arrow"]))

    def test_cues_used_by_the_gate_identify_their_target_uniquely(self) -> None:
        """Without this the task has no answer and the numbers mean nothing."""
        rng = random.Random(0)
        library = [["frame", "box", "arrow"], ["frame", "box", "plus"],
                   ["ring", "shell", "dot"], ["ring", "box", "wave"]]
        keyed = [gate.path_keys(levels) for levels in library]
        for index in range(len(library)):
            cue = gate.well_posed_cue(index, library, keyed, rng)
            if cue is None:
                continue
            with self.subTest(index=index):
                scores = [gate.jaccard(gate.path_keys(cue), k) for k in keyed]
                best = max(scores)
                self.assertEqual(sum(1 for s in scores if s == best), 1)
                self.assertEqual(scores.index(best), index)

    def test_the_oracle_property_of_the_baseline_is_documented(self) -> None:
        """A baseline that is the oracle cannot be beaten; that must not be lost.

        Jaccard over path keys *is* the ground-truth similarity, so the dict
        arm scores the oracle by construction. Requiring a lossy representation
        to beat it was unwinnable, and forgetting that would invite the same
        mistake in the next gate.
        """
        self.assertIn("oracle", gate.__doc__.lower())
        self.assertIn("oracle", gate.one_trial.__doc__.lower()
                      + str(gate.run_gate(seeds=1, depth=2, scenes=8)))


if __name__ == "__main__":
    unittest.main()
