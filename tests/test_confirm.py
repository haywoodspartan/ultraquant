"""A "yes" confirms what was asserted, not what is held now.

11.119: the confirmation and consolidation slots stored a key, and
"yes" acted on whatever that key held by then - so a command between
the assertion and the "yes" could make the user's word vouch for a
belief they never saw. These pins hold the snapshot, the arm, and the
gate's record.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T
from ultraquant.interpreter.chat import ChatCLI


class _Chat:
    def __init__(self, snapshot_on: bool = True) -> None:
        self.previous = T._CONFIRM_SNAPSHOT
        T._CONFIRM_SNAPSHOT = snapshot_on
        self.root = Path(tempfile.mkdtemp(prefix="uq_confirm_t_"))
        self.session = T.build_session(self.root, seed=0)
        self.out = io.StringIO()
        self.cli = ChatCLI(self.session, out=self.out)

    def say(self, line: str) -> str:
        mark = self.out.tell()
        self.cli.handle(line, iter([]))
        return self.out.getvalue()[mark:].strip()

    def learn_answer(self, subject: str, answer: str) -> None:
        self.say(":learn")
        for _ in range(12):
            question = self.cli.learning.next_question()
            if question is None or question.subject == subject:
                break
            self.say(":learn skip")
        self.say(f":learn answer {answer}")

    def close(self) -> None:
        T._CONFIRM_SNAPSHOT = self.previous
        shutil.rmtree(self.root, ignore_errors=True)


class ConfirmSnapshotTests(unittest.TestCase):

    def setUp(self) -> None:
        self.chat = _Chat()

    def tearDown(self) -> None:
        self.chat.close()

    def test_a_changed_value_is_not_confirmed(self) -> None:
        self.chat.say("the tower material is iron")
        self.chat.say("is the tower material iron?")
        self.chat.learn_answer("tower material", "steel")
        reply = self.chat.say("yes")
        self.assertIn("That changed since I said it", reply)
        held = self.chat.session.memory.recall_fact("tower material")
        self.assertLess(held["confidence"], 0.9)

    def test_a_placeholder_is_never_confirmed(self) -> None:
        self.chat.say("the gate colour is red")
        self.chat.say("is the gate colour red?")
        self.chat.learn_answer("gate colour", "no")
        self.assertIn("Nothing was confirmed", self.chat.say("yes"))

    def test_a_moved_premise_is_not_consolidated(self) -> None:
        for line in ("the tower material is steel",
                     "the steel hardness is high",
                     "what is the tower hardness?"):
            self.chat.say(line)
        self.chat.learn_answer("steel hardness", "low")
        reply = self.chat.say("yes")
        self.assertIn("That premise changed since I said it", reply)
        self.assertIsNone(
            self.chat.session.memory.recall_fact("tower hardness"))

    def test_an_immediate_yes_still_confirms(self) -> None:
        self.chat.say("the tower material is iron")
        self.chat.say("is the tower material iron?")
        self.assertTrue(self.chat.say("yes").startswith("Confirmed:"))


class ArmTests(unittest.TestCase):

    def test_the_flag_restores_the_stale_confirmation(self) -> None:
        chat = _Chat(snapshot_on=False)
        try:
            chat.say("the tower material is iron")
            chat.say("is the tower material iron?")
            chat.learn_answer("tower material", "steel")
            self.assertIn("Confirmed: tower material is steel",
                          chat.say("yes"))
        finally:
            chat.close()

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(T._CONFIRM_SNAPSHOT)


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import confirm_gate
        self.gate = confirm_gate
        self.doc = " ".join(confirm_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("No stale confirmation",
                       "Immediate confirmations unchanged",
                       "One-turn freshness preserved",
                       "Nothing else moves", "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_battery_can_fail(self) -> None:
        self.assertIn("The OLD arm must confirm the changed belief in all "
                      "six", self.doc)
        self.assertEqual(len(self.gate.STORED) + len(self.gate.DERIVED), 6)

    def test_the_result_and_its_cost_are_recorded(self) -> None:
        self.assertIn("PASSED on all four measured criteria", self.doc)
        self.assertIn("16.22 -> 16.34", self.doc)


if __name__ == "__main__":
    unittest.main()
