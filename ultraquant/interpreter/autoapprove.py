"""Automatic approval of factual claims, with a durable, exact undo record."""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ultraquant.interpreter.stash import (
    _independent_sources, _split_claim, claim_relation,
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


class AutoApprover:
    def __init__(self, stash, memory, journal_path) -> None:
        self.stash = stash
        self.memory = memory
        self.journal_path = Path(journal_path)

    def approve_all(self) -> list[Approval]:
        """Analyze, reject malformed claims, then approve eligible entries."""
        # Analysis calls confirm_fact for matching claims. Preserve those
        # records before analysis so its incidental reinforcement cannot
        # escape the approval's undo record (or reinforce an ineligible claim).
        analysis_before = {}
        for entry in self.stash.entries():
            if entry["status"] in ("promoted", "rejected"):
                continue
            split = _split_claim(entry["claim"], structured="provenance" in entry)
            if split is not None:
                key = split[0]
                analysis_before[key] = copy.deepcopy(self.memory._fact_record(key))
        try:
            self.stash.analyze(self.memory)
        finally:
            for key, record in analysis_before.items():
                if self.memory._fact_record(key) != record:
                    self.memory.restore_fact(key, record)

        approved = []
        for entry in sorted(self.stash.entries(), key=lambda e: e["id"]):
            if entry["status"] in ("promoted", "rejected"):
                continue
            reason = self._malformed(entry)
            if reason is not None:
                self.stash.reject(entry["id"], f"malformed: {reason}")
            elif self._eligible(entry):
                key, value = _split_claim(entry["claim"], structured="provenance" in entry)
                before = self._snapshot(key)
                self.stash.promote(
                    entry["id"], self.memory,
                    force=(entry["status"] == "disputed"
                           or entry["classification"] != "factual-claim"),
                    confidence=self._confidence(entry),
                )
                result = self.stash.last_promotion
                approval = Approval(
                    approval_id=uuid.uuid4().hex, entry_id=entry["id"],
                    key=key, value=value, confidence=result["confidence"],
                    sources=copy.deepcopy(entry["sources"]), time=time.time(),
                    before=before, outcome=result["outcome"],
                )
                self._journal({"event": "approval", **asdict(approval)})
                approved.append(approval)
        return approved

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
        split = _split_claim(claim, structured="provenance" in entry)
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

    def _restore(self, before) -> None:
        for key, record in before.items():
            self.memory.restore_fact(key, record)

    def _confidence(self, entry) -> float | None:
        return entry.get("measured_confidence")

    def dispute(self, target, reason) -> Approval:
        """Undo the latest undisputed approval for a fact key or stash id."""
        approval = next((a for a in reversed(self.approvals())
                         if not a.disputed and
                         (a.entry_id == target if isinstance(target, int)
                          else a.key == target)), None)
        if approval is None:
            raise KeyError(target)
        # Remove conclusions made since approval before restoring the old
        # chain. Both traversal and writes use the active memory backing.
        self.memory._retract_derivatives(approval.key)
        self._restore(approval.before)
        self.stash.reject(approval.entry_id, f"disputed: {reason}")
        self.memory.remember_episode(
            "dispute", {"key": approval.key, "value": approval.value,
                        "reason": reason, "restored": copy.deepcopy(approval.before)},
            tags=["fact", approval.key],
        )
        approval.disputed = True
        approval.dispute_reason = reason
        self._journal({"event": "dispute", "approval_id": approval.approval_id,
                       "reason": reason, "time": time.time()})
        return approval

    def approvals(self) -> list[Approval]:
        """Rebuild approvals and their dispute status from the journal."""
        if not self.journal_path.exists():
            return []
        approvals = {}
        with self.journal_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                event = row.pop("event", "approval")
                if event == "approval":
                    approval = Approval(**row)
                    approvals[approval.approval_id] = approval
                elif event == "dispute":
                    approval = approvals[row["approval_id"]]
                    approval.disputed = True
                    approval.dispute_reason = row["reason"]
        return list(approvals.values())

    def _journal(self, row) -> None:
        payload = json.dumps(row, ensure_ascii=False) + "\n"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
