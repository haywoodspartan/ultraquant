"""The blackboard: several partial results held at once.

The reasoning step was `one input -> one expert -> one label`, which cannot
represent "an arrow inside a frame" — not for want of experts but for want of
anywhere to put two answers. These tests cover the workspace itself and the two
things that make it more than a loop over experts: contributions filed against
*aspects* rather than the whole input, and later rounds that can revise earlier
claims.

The composition gate's verdict lives in ARCHITECTURE.md §11.6, not here. These
tests must not start failing because a measured number moved.
"""

from __future__ import annotations

import random
import unittest

from ultraquant.experiments import composition_gate as gate
from ultraquant.reason.blackboard import (
    Blackboard,
    ConstraintContributor,
    Contributor,
    ExpertContributor,
    compose,
    run_blackboard,
)


class _Fixed:
    """A contributor that posts one thing, once."""

    def __init__(self, name: str, slot: str, value: str, confidence: float) -> None:
        self.name = name
        self.slot = slot
        self.value = value
        self.confidence = confidence
        self.calls = 0

    def applies(self, board: Blackboard) -> bool:
        return not any(e.source == self.name for e in board.contributions())

    def contribute(self, board: Blackboard) -> bool:
        self.calls += 1
        board.post(self.name, self.slot, self.value, self.confidence)
        return True


class BlackboardTests(unittest.TestCase):
    """The workspace."""

    def setUp(self) -> None:
        self.board = Blackboard([0.5] * 4, meta={"origin": "test"})

    def test_different_slots_compose_rather_than_compete(self) -> None:
        self.board.post("a", "shape", "box", 0.9)
        self.board.post("b", "mark", "plus", 0.8)
        self.assertEqual(self.board.reading(), {"shape": "box", "mark": "plus"})

    def test_same_slot_competes_and_the_strongest_wins(self) -> None:
        self.board.post("a", "shape", "box", 0.6)
        self.board.post("b", "shape", "rails", 0.9)
        self.assertEqual(self.board.reading(), {"shape": "rails"})

    def test_a_later_round_wins_an_exact_tie(self) -> None:
        """A revision that merely matches is still the better-informed claim."""
        self.board.post("a", "shape", "box", 0.7)
        self.board.round = 2
        self.board.post("b", "shape", "rails", 0.7)
        self.assertEqual(self.board.best("shape").value, "rails")

    def test_joint_confidence_is_the_weakest_link(self) -> None:
        """Averaging would let a sure slot disguise a coin flip on another."""
        self.board.post("a", "shape", "box", 0.99)
        self.board.post("b", "mark", "plus", 0.34)
        self.assertAlmostEqual(self.board.confidence(), 0.34)

    def test_an_empty_board_reads_as_nothing(self) -> None:
        self.assertEqual(self.board.reading(), {})
        self.assertEqual(self.board.confidence(), 0.0)
        self.assertEqual(self.board.slots(), [])

    def test_snapshot_is_json_safe_and_complete(self) -> None:
        import json

        self.board.post("a", "shape", "box", 0.9, {"why": "border"})
        snapshot = self.board.snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        self.assertEqual(len(snapshot["contributions"]), 1)
        self.assertEqual(snapshot["reading"], {"shape": "box"})


class ControllerTests(unittest.TestCase):
    """Running contributors to quiescence."""

    def test_it_stops_once_nothing_new_is_written(self) -> None:
        one = _Fixed("one", "shape", "box", 0.9)
        two = _Fixed("two", "mark", "plus", 0.8)
        board = compose([0.0], [one, two], max_rounds=9)
        self.assertEqual((one.calls, two.calls), (1, 1))
        self.assertEqual(board.round, 2, "one working round, one to notice quiet")

    def test_a_contributor_that_declines_writes_nothing(self) -> None:
        class Silent:
            name = "silent"

            def applies(self, board):
                return False

            def contribute(self, board):  # pragma: no cover - never called
                raise AssertionError("should not be asked to contribute")

        self.assertEqual(compose([0.0], [Silent()]).reading(), {})

    def test_max_rounds_bounds_a_pair_that_keeps_revising(self) -> None:
        """Two contributors undoing each other must not spin forever."""

        class Flip:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def applies(self, board):
                best = board.best("shape")
                return best is None or best.value != self.value

            def contribute(self, board):
                board.post(self.name, "shape", self.value, 0.9)
                return True

        board = compose([0.0], [Flip("a", "x"), Flip("b", "y")], max_rounds=3)
        self.assertEqual(board.round, 3)

    def test_the_protocol_is_satisfied_by_the_real_contributors(self) -> None:
        constraint = ConstraintContributor("c", lambda r: True, {"shape": ["box"]})
        self.assertIsInstance(constraint, Contributor)


class ConstraintTests(unittest.TestCase):
    """The part a plain loop over experts cannot do."""

    def _board(self, shape_conf: float, mark_conf: float) -> Blackboard:
        board = Blackboard([0.0])
        board.post("expert:shape", "shape", "box", shape_conf)
        board.post("expert:mark", "mark", "dot", mark_conf)
        return board

    def _rule(self):
        """'box' never appears with 'dot'."""
        return ConstraintContributor(
            "rule",
            allowed=lambda r: not (r.get("shape") == "box" and r.get("mark") == "dot"),
            options={"shape": ["box", "rails"], "mark": ["dot", "plus"]},
        )

    def test_it_repairs_the_weakest_slot_not_the_strongest(self) -> None:
        board = self._board(shape_conf=0.95, mark_conf=0.40)
        run_blackboard(board, [self._rule()])
        reading = board.reading()
        self.assertEqual(reading["shape"], "box", "the confident slot survives")
        self.assertEqual(reading["mark"], "plus", "the weak slot was corrected")

    def test_it_repairs_the_other_way_when_the_weakness_moves(self) -> None:
        board = self._board(shape_conf=0.40, mark_conf=0.95)
        run_blackboard(board, [self._rule()])
        reading = board.reading()
        self.assertEqual(reading["mark"], "dot")
        self.assertEqual(reading["shape"], "rails")

    def test_a_consistent_reading_is_left_alone(self) -> None:
        board = Blackboard([0.0])
        board.post("expert:shape", "shape", "rails", 0.6)
        board.post("expert:mark", "mark", "dot", 0.5)
        run_blackboard(board, [self._rule()])
        self.assertEqual(board.reading(), {"shape": "rails", "mark": "dot"})
        self.assertEqual(len(board.contributions()), 2, "nothing was written")

    def test_it_needs_the_other_results_to_be_there_first(self) -> None:
        """This is the dependency that makes rounds necessary."""
        board = Blackboard([0.0])
        board.post("expert:shape", "shape", "box", 0.9)
        self.assertFalse(self._rule().applies(board), "mark is not filled in yet")


class ExpertContributorTests(unittest.TestCase):
    """Wrapping a real library expert."""

    class _Pool:
        def __init__(self, answers):
            self.answers = answers
            self.seen = []

        def has_expert(self, category):
            return category in self.answers

        def predict(self, category, features):
            self.seen.append((category, tuple(features)))
            return self.answers[category]

    def test_it_posts_the_expert_reading(self) -> None:
        pool = self._Pool({"shape": ("box", 0.87)})
        board = compose([1.0, 2.0], [ExpertContributor(pool, "shape", "shape")])
        self.assertEqual(board.reading(), {"shape": "box"})
        self.assertAlmostEqual(board.best("shape").confidence, 0.87)

    def test_every_expert_sees_the_whole_input(self) -> None:
        """Slicing the input per expert would do the decomposition by hand."""
        pool = self._Pool({"shape": ("box", 0.9), "mark": ("plus", 0.8)})
        features = [0.1, 0.2, 0.3]
        compose(features, [ExpertContributor(pool, "shape", "shape"),
                           ExpertContributor(pool, "mark", "mark")])
        self.assertEqual([f for _c, f in pool.seen], [tuple(features)] * 2)

    def test_a_missing_expert_declines(self) -> None:
        pool = self._Pool({})
        self.assertEqual(compose([0.0], [ExpertContributor(pool, "gone", "s")]).reading(), {})

    def test_a_low_confidence_reading_can_be_withheld(self) -> None:
        pool = self._Pool({"shape": ("box", 0.2)})
        contributor = ExpertContributor(pool, "shape", "shape", min_confidence=0.5)
        self.assertEqual(compose([0.0], [contributor]).reading(), {})


class CompositionTaskTests(unittest.TestCase):
    """The gate's task is well-formed, independent of the result it produced."""

    def test_shapes_and_marks_never_overlap(self) -> None:
        """Overlapping factors would make the task ambiguous, not compositional."""
        for shape, border in gate.SHAPES.items():
            for mark, interior in gate.MARKS.items():
                with self.subTest(shape=shape, mark=mark):
                    self.assertEqual(set(border) & set(interior), set())

    def test_every_part_of_a_held_out_pair_was_seen_in_training(self) -> None:
        """Otherwise it tests recognition of the unseen, not composition."""
        for seed in range(8):
            with self.subTest(seed=seed):
                seen, unseen = gate.split_combinations(random.Random(seed))
                seen_shapes = {s for s, _ in seen}
                seen_marks = {m for _, m in seen}
                for shape, mark in unseen:
                    self.assertIn(shape, seen_shapes)
                    self.assertIn(mark, seen_marks)

    def test_held_out_pairs_never_appear_in_training(self) -> None:
        seen, unseen = gate.split_combinations(random.Random(0))
        self.assertEqual(set(seen) & set(unseen), set())
        self.assertEqual(len(seen) + len(unseen), len(gate.SHAPES) * len(gate.MARKS))

    def test_the_primary_vocabulary_is_nested_as_documented(self) -> None:
        """The flaw is recorded, not quietly fixed after the fact.

        ``dot`` is inside ``bar`` is inside ``plus``, which caps how well the
        mark factor can be read at all. The gate was measured on this, so it
        stays; the claim about it needs to keep matching the data.
        """
        marks = gate.MARKS
        self.assertLess(set(marks["dot"]), set(marks["bar"]))
        self.assertLess(set(marks["bar"]), set(marks["plus"]))

    def test_the_distinct_vocabulary_removes_the_nesting(self) -> None:
        """The secondary condition must not repeat the nesting flaw.

        Non-containment is the property that matters. Distance 6 is impossible
        for four patterns on the inner 3x3 — it holds only three disjoint
        triples — so 4 is the achievable bar, not a lowered one.
        """
        items = list(gate.DISTINCT_MARKS.items())
        for i, (name_a, a) in enumerate(items):
            for name_b, b in items[i + 1:]:
                with self.subTest(pair=(name_a, name_b)):
                    self.assertFalse(set(a) <= set(b), "one mark contains another")
                    self.assertFalse(set(b) <= set(a), "one mark contains another")
                    self.assertGreaterEqual(len(set(a) ^ set(b)), 4)


class ComposedReadingThroughTheInterpreterTests(unittest.TestCase):
    """Stage 2 delivered into the system, not only demonstrated in a script.

    A library whose categories describe *different aspects* of an input has its
    experts composed; a library whose categories are alternatives has them
    compete, which is what every library built before slots existed does. The
    paging discipline is the thing to watch: one expert per aspect of the answer
    and never more.
    """

    def setUp(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        from ultraquant.experts.moe import ExpertPool
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_composed_"))
        self.session = build_session(self.dir / "home", seed=0)
        rng = random.Random(2)
        self.seen, self.unseen = gate.split_combinations(rng, held_out=6)
        train_x, train_y = gate._samples(rng, self.seen, 24, 2)

        shapes, marks = sorted(gate.SHAPES), sorted(gate.MARKS)
        self.session.experts.train_expert(
            "border", train_x, [shapes.index(s) for s, _ in train_y], shapes,
            epochs=40, lr=0.05)
        self.session.experts.train_expert(
            "inner", train_x, [marks.index(m) for _, m in train_y], marks,
            epochs=40, lr=0.05)
        prototypes = [gate.compound(s, m) for s, m in self.seen]
        for category, slot in (("border", "shape"), ("inner", "mark")):
            shard = ExpertPool.shard_id(category)
            self.session.vault.set_slot(shard, slot)
            self.session.vault.set_signature(shard, prototypes)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _rows(pixels: list[float]) -> list[str]:
        return ["".join("#" if pixels[r * 5 + c] else "." for c in range(5))
                for r in range(5)]

    def _ask(self, shape: str, mark: str):
        from ultraquant.interpreter.thoughts import run_pipeline

        return run_pipeline("\n".join(self._rows(gate.compound(shape, mark))),
                            self.session)

    def test_the_library_declares_two_aspects(self) -> None:
        self.assertEqual(self.session.vault.slots(),
                         {"border": "shape", "inner": "mark"})

    def test_a_compound_input_gets_a_compound_reading(self) -> None:
        response, _trace = self._ask(*self.seen[0])
        self.assertIn("shape:", response)
        self.assertIn("mark:", response)

    def test_both_factors_are_filed_against_their_own_slots(self) -> None:
        _response, _trace = self._ask(*self.seen[0])
        # data survives on the session's last trace via the pipeline context
        board = self.session.last_trace
        self.assertTrue(board, "the pipeline recorded no trace")

    def test_combinations_never_seen_together_are_still_read(self) -> None:
        """The whole point: parts were learned, the pairing was not."""
        correct = 0
        for shape, mark in self.unseen:
            response, _trace = self._ask(shape, mark)
            correct += shape in response and mark in response
        self.assertGreaterEqual(
            correct, 3,
            "composition should read most unseen combinations; a monolithic "
            "classifier scores exactly zero on these",
        )

    def test_one_expert_is_paged_per_aspect_and_no_more(self) -> None:
        self._ask(*self.seen[0])
        resident = sorted(self.session.cache.stats()["resident"])
        self.assertEqual(resident, ["expert:border", "expert:inner"])

    def test_a_single_slot_library_still_pages_only_one_expert(self) -> None:
        """Libraries that never declare slots must pay exactly what they did."""
        from ultraquant.experts.moe import ExpertPool
        from ultraquant.interpreter.thoughts import build_session

        session = build_session(self.dir / "home", seed=0)
        for category in ("border", "inner"):
            session.vault.set_slot(ExpertPool.shard_id(category), "pattern")
        run = self._rows(gate.compound(*self.seen[0]))
        from ultraquant.interpreter.thoughts import run_pipeline

        run_pipeline("\n".join(run), session)
        self.assertEqual(len(session.cache.stats()["resident"]), 1)


if __name__ == "__main__":
    unittest.main()
