"""Automatic approval of factual claims, with a durable, exact undo record.

Undo checks per-key after states, conservative legacy identity, and the
premises of restored records. Recorded losses keep arbitration settled
across passes; rollback restores those entries and removes its episodes.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ultraquant.interpreter.stash import (
    _claim_provenance, _independent_sources, claim_relation,
)


@dataclass
class Approval:
    approval_id: str
    entry_id: int
    key: str
    value: Any
    confidence: float
    sources: list
    time: float
    before: dict
    outcome: str
    disputed: bool = False
    dispute_reason: str = ""
    after: dict | None = None
    after_states: dict | None = None


# §11.139: catalogue metadata is never evidence of a changed belief.
def _belief_record(record):
    """The record identity used by approvals, without catalogue fields."""
    return ({key: value for key, value in record.items()
             if key not in ("subject", "attribute")} if record is not None else None)


class AutoApprover:
    def __init__(self, stash, memory, journal_path) -> None:
        self.stash = stash
        self.memory = memory
        self.journal_path = Path(journal_path)
        self._batch_approved = set()
        self.recover()

    def approve_all(self) -> list[Approval]:
        """Analyze, reject malformed claims, then approve eligible entries."""
        self.recover()
        # Classification must not reinforce facts outside a transaction.
        self.stash.analyze(self.memory, reinforce=False)
        rejected = {_claim_provenance(e) for e in self.stash.entries()
                    if e["status"] == "rejected"}
        rejected.discard(None)

        approved = []
        self._batch_approved = set()
        for entry in sorted(self.stash.entries(), key=lambda e: e["id"]):
            # Review 6: a previous winner may have rejected this entry.
            entry = self.stash.get(entry["id"])
            if entry["status"] in ("promoted", "rejected"):
                continue
            if _claim_provenance(entry) in rejected:
                continue
            reason = self._malformed(entry)
            if reason is not None:
                self.stash.reject(entry["id"], f"malformed: {reason}")
                provenance = _claim_provenance(entry)
                if provenance is not None:
                    rejected.add(provenance)
            elif self._eligible(entry):
                key, value = self.stash._split_entry(entry)
                before = self._snapshot(key)
                transaction_id = uuid.uuid4().hex
                confidence = self._confidence(entry)
                count = len(_independent_sources(entry["sources"]))
                losers = [copy.deepcopy(other) for other in self.stash.entries()
                          if other["id"] != entry["id"]
                          and other.get("classification") == "factual-claim"
                          and other["status"] in ("staged", "corroborated", "disputed")
                          and claim_relation(entry["claim"], other["claim"]) == "contradicts"
                          and len(_independent_sources(other["sources"])) < count]
                episode_start = self.memory._next_id
                self._journal({"event": "intent", "operation": "approval",
                               "transaction_id": transaction_id,
                               "entry_id": entry["id"], "key": key,
                               "value": value, "sources": entry["sources"],
                               "confidence": confidence, "time": time.time(),
                               "before": before,
                               "losers": losers, "episode_start": episode_start,
                               "entry": copy.deepcopy(entry)})
                try:
                    self.stash.promote(
                        entry["id"], self.memory,
                        force=(entry["status"] == "disputed"
                               or entry["classification"] != "factual-claim"),
                        confidence=confidence,
                    )
                    # Review 6: rejection, journalled with the winner, is
                    # what prevents a defeated rival winning a later pass.
                    for loser in losers:
                        self._record_loss(loser, entry)
                finally:
                    # Include revision/retraction episodes, even if promotion
                    # failed partway through and recovery runs in this process.
                    for episode in self.memory._episodes:
                        if episode["id"] >= episode_start:
                            episode["tags"].append(f"transaction:{transaction_id}")
                result = self.stash.last_promotion
                approval = Approval(
                    approval_id=transaction_id, entry_id=entry["id"],
                    key=key, value=value, confidence=result["confidence"],
                    sources=copy.deepcopy(entry["sources"]), time=time.time(),
                    before=before, outcome=result["outcome"],
                    after=copy.deepcopy(self.memory._fact_record(key)),
                    after_states={name: copy.deepcopy(self.memory._fact_record(name))
                                  for name in before},
                )
                self._persist()
                self._journal({"event": "commit", "operation": "approval",
                               "transaction_id": transaction_id,
                               **asdict(approval)})
                approved.append(approval)
                self._batch_approved.add(entry["id"])
        return approved

    def _record_loss(self, entry, winner) -> None:
        self.stash.reject(entry["id"], f"lost to entry {winner['id']}")

    def _eligible(self, entry) -> bool:
        if (entry.get("classification") != "factual-claim"
                or entry.get("status") not in ("staged", "corroborated", "disputed")):
            return False
        if entry["status"] == "disputed":
            count = len(_independent_sources(entry["sources"]))
            for other in self.stash.entries():
                if (other["id"] == entry["id"] or other["status"] == "rejected"
                        or other["classification"] != "factual-claim"):
                    continue
                if (other["status"] == "promoted"
                        and other["id"] not in self._batch_approved):
                    # An earlier batch is the memory being revised. Only
                    # this batch's winner can block its losing rival.
                    continue
                # Include an already promoted rival: id order must not let
                # the losing side win after the better-sourced side is filed.
                # Re-analysis can relabel that clash as a memory dispute.
                if (claim_relation(entry["claim"], other["claim"]) == "contradicts"
                        and count <= len(_independent_sources(other["sources"]))):
                    return False
        return True

    def _malformed(self, entry) -> str | None:
        if entry.get("classification") != "factual-claim":
            return None
        claim = entry["claim"]
        split = self.stash._split_entry(entry)
        if split is None:
            return "no subject and value"
        if split[0].split()[0] in {"what", "who", "where", "when", "which", "how", "why"}:
            return "question instead of a claim"
        compact = re.sub(r"\s+", "", claim.casefold())
        if any(mark in compact for mark in (
                "endturntoken", "<|", "|>", "im_start", "im_end",
                "[inst]", "[/inst]", "end_of_turn")):
            return "chat-template residue"
        if compact.startswith(("youhavementioned", "iholdnothing")):
            return "system chatter"
        return None

    def _snapshot(self, key) -> dict:
        return {name: copy.deepcopy(self.memory._fact_record(name))
                for name in [key, *self.memory.derivatives_of(key)]}

    # review 8: catalogue structure describes the key across value changes.
    def _with_catalogue(self, key, record):
        if record is not None and ("subject" not in record or "attribute" not in record):
            current = self.memory._fact_record(key)
            additions = {field: current[field] for field in ("subject", "attribute")
                         if field not in record and current is not None and field in current}
            if additions:
                return dict(record, **additions)
        return record

    def _restore(self, before) -> dict:
        # The snapshot starts with the approved key. Restore it first, then
        # restore conclusions in dependency order, not dictionary order.
        pending = dict(before)
        if not pending:
            return {}
        key = next(iter(pending))
        record = pending.pop(key)
        dropped = {}
        # Review 6: the primary has premises too; stale ones stay absent.
        if record is None or self._restorable(key, record):
            # review 8: dispute restores keep structure learned after approval.
            self.memory.restore_fact(key, self._with_catalogue(key, record))
        else:
            self.memory.restore_fact(key, self._with_catalogue(key, None))
            dropped[key] = record
        while pending:
            ready = [key for key, record in pending.items()
                     if record is None or self._restorable(key, record)]
            if not ready:
                break
            for key in ready:
                # review 8: secondary restores and replay use the same rule.
                self.memory.restore_fact(key, self._with_catalogue(key, pending.pop(key)))
        dropped.update(pending)
        # Use the same premise predicate for the explanation as for undo.
        return {key: [p_key for p_key, p_value in record.get("derived_from", [])
                      if not self._restorable(key, {"derived_from": [(p_key, p_value)]})]
                for key, record in dropped.items()}

    def _still_holds(self, key, expected) -> bool:
        # §11.139: migration does not supersede a secondary undo state.
        return _belief_record(self.memory._fact_record(key)) == _belief_record(expected)

    def _journalled_descendants(self, intent) -> list:
        # review 7: replay must also reach descendants whose parent is gone.
        return list(intent.get("descendants") or [])

    def _drop_stale_derivatives(self, key, descendants) -> None:
        """Truth maintenance after an undo: drop what now rests on nothing.

        Claude's review of §11.136: sparing every changed secondary left
        "safety: unsafe", derived from a height of 200, standing after the
        height went back to 100. Retracting every derivative instead dropped
        a conclusion the user had confirmed whose premise the undo never
        changed. So each derivative of the disputed key is judged by its
        own premises, after the restore, until nothing else falls.
        """
        # review 7: retain the journalled keys even after the graph is broken.
        names = dict.fromkeys(descendants)
        dropped = True
        while dropped:
            dropped = False
            names.update(dict.fromkeys(self.memory.derivatives_of(key)))
            for name in names:
                record = self.memory._fact_record(name)
                if record is not None and not self._restorable(name, record):
                    # review 8: all dispute restores share the catalogue seam.
                    self.memory.restore_fact(name, self._with_catalogue(name, None))
                    dropped = True

    def _restore_plan(self, approval) -> list:
        """The secondary keys an exact undo may restore, judged right now.

        Review 6: a secondary comes back only while it still holds the state
        the approval left it in. Journals without per-key states imply it:
        a revision retracted every derivative, anything else left it as is.
        """
        plan = []
        for key, record in approval.before.items():
            if key == approval.key:
                continue
            if approval.after_states is not None and key in approval.after_states:
                expected = approval.after_states[key]
            else:
                expected = None if approval.outcome == "revised" else record
            if self._still_holds(key, expected):
                plan.append(key)
        return plan

    def _legacy_untouched(self, approval, current) -> bool:
        # Review 6: equal values alone cannot identify a legacy approval.
        if not (current is not None
                and approval.outcome == "new"
                and "derived_from" not in current
                and current["value"] == approval.value
                and not current.get("negated", False)
                and current["confidence"] == approval.confidence
                and current.get("reinforcements", 0) == 0
                and current.get("last_seen") == current.get("first_seen")):
            return False
        # review 7: a later creation cannot be the approval's own record.
        try:
            return datetime.fromisoformat(current.get("first_seen")).timestamp() <= approval.time
        except (TypeError, ValueError, OverflowError, OSError):
            return False

    def _restorable(self, key, record) -> bool:
        for premise_key, premise_value in record.get("derived_from", []):
            current = self.memory._fact_record(premise_key)
            if current is None:
                return False
            value = current["value"]
            if current.get("negated", False):
                value = f"not {value}"
            if str(value) != str(premise_value):
                return False
        return True

    def _dispute_mode(self, approval) -> str:
        later = False
        for other in self.approvals():
            if later and other.key == approval.key and not other.disputed:
                return "refuse"
            if other.approval_id == approval.approval_id:
                later = True
        current = self.memory._fact_record(approval.key)
        if approval.after is not None:
            # §11.139: structure changes neither the approval nor its undo.
            return ("exact" if _belief_record(current) == _belief_record(approval.after)
                    else "superseded")
        return "exact" if self._legacy_untouched(approval, current) else "superseded"

    def _confidence(self, entry) -> float | None:
        return entry.get("measured_confidence")

    def dispute(self, target, reason) -> Approval:
        """Undo the latest approval for a fact key or stash id, once."""
        self.recover()
        matches = [a for a in reversed(self.approvals())
                   if (a.entry_id == target if isinstance(target, int)
                       else a.key == target)]
        if not matches:
            raise KeyError(target)
        approval = matches[0]
        if approval.disputed:
            return approval
        mode = self._dispute_mode(approval)
        if mode == "refuse":
            raise ValueError("Dispute the newest approval for this key first")
        intent = {"event": "intent", "operation": "dispute",
                  "transaction_id": uuid.uuid4().hex,
                  "approval": asdict(approval), "mode": mode,
                  "reason": reason,
                  # Judged once, before anything changes, so a replay after a
                  # crash restores exactly what the first attempt did.
                  "restore": (self._restore_plan(approval)
                              if mode == "exact" else []),
                  # review 7: capture reachability before the first change.
                  "descendants": (self.memory.derivatives_of(approval.key)
                                  if mode == "exact" else [])}
        self._journal(intent)
        self._complete_dispute(intent)
        approval.disputed = True
        approval.dispute_reason = reason
        return approval

    def _complete_dispute(self, intent) -> None:
        approval = Approval(**intent["approval"])
        reason = intent["reason"]
        changed_premises = {}
        if intent["mode"] == "exact":
            plan = intent.get("restore")
            if plan is None:            # an intent journalled before plans
                plan = self._restore_plan(approval)
            before = {approval.key: approval.before[approval.key]}
            before.update({key: approval.before[key] for key in plan})
            changed_premises = self._restore(before)
            self._drop_stale_derivatives(
                approval.key, self._journalled_descendants(intent))
        self.stash.reject(approval.entry_id, f"disputed: {reason}")
        # Replaying a dispute after persistence but before commit must not
        # duplicate its episode. Only one dispute exists per approval.
        episodes = self.memory.recall_episodes(kind="dispute", limit=float("inf"))
        if not any(e["content"].get("approval_id") == approval.approval_id
                   for e in episodes):
            self.memory.remember_episode(
                "dispute", {"key": approval.key, "value": approval.value,
                            "approval_id": approval.approval_id, "reason": reason,
                            "restored": {key: copy.deepcopy(self.memory._fact_record(key))
                                         for key in approval.before}
                            if intent["mode"] == "exact" else {},
                            "changed_premises": changed_premises,
                            "note": "later change was kept"
                            if intent["mode"] == "superseded" else
                            "approval undone; changed premises: "
                            + ", ".join(sorted({p for keys in changed_premises.values()
                                                for p in keys}))
                            if changed_premises else "approval undone"},
                tags=["fact", approval.key],
            )
        self._persist()
        self._journal({"event": "commit", "operation": "dispute",
                       "transaction_id": intent["transaction_id"],
                       "approval_id": approval.approval_id,
                       "reason": reason, "time": time.time()})

    def _persist(self) -> None:
        path = self.memory.path
        if path is not None:
            # save() normally truncates its destination. Replace a complete
            # file so recovery can always open the pre- or post-write state.
            tmp = path.with_name(path.name + ".autoapprove.tmp")
            try:
                self.memory.save(tmp)
                with tmp.open("r+b") as handle:
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            finally:
                self.memory.path = path
        elif self.memory.shards is not None:
            self.memory.shards.flush()
        self.stash.save()

    def recover(self) -> None:
        """Roll back incomplete approvals and finish incomplete disputes."""
        pending = {}
        for row in self._rows():
            if row.get("event") == "intent":
                pending[row["transaction_id"]] = row
            elif row.get("event") in ("commit", "rollback"):
                pending.pop(row["transaction_id"], None)
        for transaction_id, intent in pending.items():
            if intent["operation"] == "dispute":
                self._complete_dispute(intent)
            else:
                # Unlike a later dispute, rollback restores the entire
                # pre-transaction snapshot, including its derived records.
                self.memory._retract_derivatives(intent["key"])
                for key, record in intent["before"].items():
                    self.memory.restore_fact(key, record)
                entry = intent["entry"]
                self.stash._entries[entry["id"]] = copy.deepcopy(entry)
                for loser in intent.get("losers", []):
                    self.stash._entries[loser["id"]] = copy.deepcopy(loser)
                self._forget_transaction_episodes(transaction_id)
                self._persist()
                self._journal({"event": "rollback", "operation": "approval",
                               "transaction_id": transaction_id})

    def _forget_transaction_episodes(self, transaction_id) -> None:
        # Review 6: aborted approvals must leave no promotion history.
        self.memory.forget_episodes(f"transaction:{transaction_id}")

    def _rows(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        # An interrupted append may leave a partial last line; it was never
        # durable and must neither count as a commit nor prevent recovery.
        lines = self.journal_path.read_bytes().splitlines(keepends=True)
        return [json.loads(line) for line in lines
                if line.endswith(b"\n") and line.strip()]

    def approvals(self) -> list[Approval]:
        """Rebuild approvals and their dispute status from the journal."""
        approvals = {}
        for row in self._rows():
            event = row.pop("event", "approval")
            # Keep reading the original journal format on live libraries.
            operation = row.get("operation") if event == "commit" else event
            if operation == "approval":
                if event == "commit":
                    row.pop("operation")
                    row.pop("transaction_id")
                approval = Approval(**row)
                approvals[approval.approval_id] = approval
            elif operation == "dispute":
                approval = approvals[row["approval_id"]]
                approval.disputed = True
                approval.dispute_reason = row["reason"]
        return list(approvals.values())

    def _journal(self, row) -> None:
        payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a+b") as handle:
            handle.seek(0)
            data = handle.read()
            if data and not data.endswith(b"\n"):
                handle.truncate(data.rfind(b"\n") + 1)
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
