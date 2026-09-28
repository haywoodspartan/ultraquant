"""§11.139: structure, catalogue transactions and indexed truth maintenance."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.memory.factshards import FactShards, normalize_subject
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class CatalogueTests(unittest.TestCase):
    # §11.139: every persistence test owns a scratch library.
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="uq_catalogue_test_")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def memory(self, sharded=True, name="memory"):
        shards = (FactShards(ShardVault(self.root / (name + "_vault")))
                  if sharded else None)
        return SystematicMemory(self.root / (name + ".json"), shards=shards)

    def test_normalization_and_bucket_hash(self):
        shards = self.memory().shards
        for subject, normalized in [("  THE Brásília—州  ", "brasilia 州"),
                                    ("An Æther's  STRAẞE", "æther s strasse"),
                                    ("the a tower", "a tower")]:
            with self.subTest(subject=subject):
                self.assertEqual(normalize_subject(subject), normalized)
                digest = hashlib.blake2b(normalized.encode(), digest_size=4).digest()
                expected = f"fact:{int.from_bytes(digest, 'big') % shards.buckets:03d}"
                self.assertEqual(shards.subject_bucket(subject), expected)
        self.assertEqual(shards.bucket_of("capital of kenya"),
                         shards.subject_bucket("capital of"))

    def test_structure_survives_all_record_writes(self):
        for sharded in (False, True):
            m = self.memory(sharded, str(sharded))
            m.remember_fact("tower material", "steel", subject="The Tower",
                            attribute="material")
            m.remember_fact("tower material", "steel", subject="Tower")
            m.remember_fact("tower material", "iron", attribute="composition")
            m.confirm_fact("tower material")
            self.assertEqual((m._fact_record("tower material")["subject"],
                              m._fact_record("tower material")["attribute"]),
                             ("Tower", "composition"))
            m.consolidate_fact("tower material", "alloy", .8, [("metal", "iron")])
            self.assertEqual(m._fact_record("tower material")["subject"], "Tower")
            m.consolidate_fact("tower material", "copper", .8, [], True,
                               subject="New tower", attribute="metal")
            snapshot = copy.deepcopy(m._fact_record("tower material"))
            m.restore_fact("tower material", None)
            m.restore_fact("tower material", snapshot)
            m.save()
            m = self.memory(sharded, str(sharded))
            self.assertEqual(m._fact_record("tower material"), snapshot)
            m.remember_fact("chat fact", "unstructured")
            self.assertNotIn("subject", m._fact_record("chat fact"))

    def test_subject_lookup_matches_contiguous_normalized_ngrams(self):
        for sharded in (False, True):
            m = self.memory(sharded, str(sharded))
            for i, subject in enumerate(("The Brásília", "north tower", "tower",
                                         "one two three four five six seven eight",
                                         "one two three four five six seven eight nine")):
                m.remember_fact(str(i), "v", subject=subject, attribute="a")
            self.assertEqual(m.subjects_in("BRASILIA; north tower?"),
                             {"brasilia", "north tower", "tower"})
            self.assertEqual(m.subjects_in("northern towers"), set())
            self.assertEqual(m.subjects_in("one two three four five six seven eight nine"),
                             {"one two three four five six seven eight"})
            m.save()
            fresh = self.memory(sharded, str(sharded))
            self.assertEqual(fresh.subjects_in("north tower"), {"north tower", "tower"})

    def test_structure_moves_key_once_and_populates_indexes(self):
        m = self.memory()
        key = "capital of kenya"
        m.remember_fact(key, "Nairobi")
        m.save()
        before = m.shards.bucket_of(key)
        record = m.recall_fact(key)
        record.update(subject="Kenya", attribute="capital")
        m.restore_fact(key, record)
        after = m.shards.subject_bucket("Kenya")
        self.assertNotEqual(before, after)
        self.assertEqual(m.fact_keys(), [key])
        m.save()
        m = self.memory()
        vault = m.shards.vault
        self.assertNotIn(key, vault.get(before)["facts"])
        self.assertEqual(vault.get(after)["facts"][key], record)
        directory, subjects, attributes = {}, {}, {}
        for entry in vault.catalog():
            if entry["kind"] == "fact-index":
                payload = vault.get(entry["shard_id"])
                directory.update(payload.get("keys", {}))
                subjects.update(payload.get("subjects", {}))
                attributes.update(payload.get("attributes", {}))
        self.assertEqual(directory[key], after)
        self.assertEqual(subjects["kenya"], {"name": "Kenya", "bucket": after,
                                              "keys": [key]})
        self.assertEqual(attributes["capital"], {"name": "capital", "subjects": 1})
        m.restore_fact(key, None)
        m.save()
        self.assertEqual(self.memory().fact_keys(), [])
        self.assertEqual(self.memory().subjects_in("Kenya"), set())

    def test_attribute_counts_distinct_subjects_and_removals(self):
        m = self.memory()
        for key, subject in [("x", "Kenya"), ("y", "the KENYA"), ("z", "Ghana")]:
            m.remember_fact(key, "v", subject=subject, attribute="The Capital")
        m.save()
        self.assertEqual(m.shards.vault.get("index:attributes")["attributes"]
                         ["capital"]["subjects"], 2)
        m.restore_fact("x", None)
        m.remember_fact("y", "v", attribute="population")
        m.save()
        attrs = m.shards.vault.get("index:attributes")["attributes"]
        self.assertEqual({a: v["subjects"] for a, v in attrs.items()},
                         {"capital": 1, "population": 1})
        m.restore_fact("z", None)
        m.save()
        self.assertNotIn("capital", m.shards.vault.get("index:attributes")["attributes"])

    def test_legacy_duplicates_are_removed_on_put_and_delete(self):
        vault = ShardVault(self.root / "memory_vault")
        fs = FactShards(vault)
        key = "capital of kenya"
        record = {"value": "Nairobi", "confidence": .8}
        for bucket in {fs.bucket_of(key), fs._legacy_bucket_of(key), "fact:999"}:
            vault.add_shard(bucket, bucket, {"facts": {key: record}}, kind="fact-bucket")
        m = self.memory()
        self.assertEqual(m.fact_keys(), [key])
        m.restore_fact(key, dict(record, subject="Kenya", attribute="capital"))
        m.save()
        occupied = [e["shard_id"] for e in m.shards.vault.catalog()
                    if e["kind"] == "fact-bucket"
                    and key in m.shards.vault.get(e["shard_id"])["facts"]]
        self.assertEqual(occupied, [m.shards.subject_bucket("Kenya")])
        self.assertTrue(m.shards.delete(key))
        self.assertFalse(m.shards.delete(key))
        m.save()
        self.assertIsNone(self.memory().recall_fact(key))

    def test_failed_index_flush_is_atomic_and_retryable(self):
        m = self.memory()
        m.remember_fact("capital of kenya", "Nairobi")
        m.save()
        record = m.recall_fact("capital of kenya")
        record.update(subject="Kenya", attribute="capital")
        m.restore_fact("capital of kenya", record)
        m.consolidate_fact("conclusion", "v", .7, [("capital of kenya", "Nairobi")])
        real = m.shards.vault.add_shard

        def fail_index(*args, **kwargs):
            if kwargs.get("kind") == "fact-index":
                raise OSError("injected index failure")
            return real(*args, **kwargs)

        with mock.patch.object(m.shards.vault, "add_shard", side_effect=fail_index):
            with self.assertRaises(OSError):
                m.save()
        fresh = self.memory()
        self.assertNotIn("subject", fresh.recall_fact("capital of kenya"))
        self.assertEqual(fresh.derivatives_of("capital of kenya"), [])
        self.assertEqual(fresh.subjects_in("Kenya"), set())
        m.save()
        fresh = self.memory()
        self.assertEqual(fresh.recall_fact("capital of kenya"), record)
        self.assertEqual(fresh.derivatives_of("capital of kenya"), ["conclusion"])

    # §11.139: old full-key copies must not supersede the current prefix copy.
    def test_legacy_duplicates_keep_current_identity_across_index_restart(self):
        vault = ShardVault(self.root / "memory_vault")
        fs = FactShards(vault)
        key = "capital of kenya"
        current = {"value": "current", "derived_from": [["new premise", "v"]]}
        obsolete = {"value": "old", "derived_from": [["old premise", "v"]]}
        for bucket, record in [(fs.bucket_of(key), current),
                               (fs._legacy_bucket_of(key), obsolete),
                               ("fact:999", obsolete)]:
            vault.add_shard(bucket, bucket, {"facts": {key: record}}, kind="fact-bucket")
        m = self.memory()
        self.assertEqual(m.recall_fact(key), current)
        self.assertEqual(m._derived_candidates("old premise"), set())
        m.save()
        m = self.memory()
        m.restore_fact(key, dict(current, subject="Kenya", attribute="capital"))
        m.save()
        locations = [e["shard_id"] for e in m.shards.vault.catalog()
                     if e["kind"] == "fact-bucket"
                     and key in m.shards.vault.get(e["shard_id"])["facts"]]
        self.assertEqual(locations, [m.shards.subject_bucket("Kenya")])

    def test_only_changed_index_shards_are_written(self):
        m = self.memory()
        m.remember_fact("capital of kenya", "Nairobi", subject="Kenya", attribute="capital")
        m.save()
        with mock.patch.object(m.shards.vault, "add_shard", wraps=m.shards.vault.add_shard) as add:
            m.confirm_fact("capital of kenya")
            m.save()
            self.assertTrue(add.called)
            self.assertFalse(any(c.kwargs.get("kind") == "fact-index" for c in add.call_args_list))
            add.reset_mock()
            m.save()
            add.assert_not_called()

    def test_derivation_closure_updates_and_survives_restart(self):
        for sharded in (False, True):
            m = self.memory(sharded, str(sharded))
            m.remember_fact("a", "v")
            m.consolidate_fact("b", "v", .7, [("a", "v")])
            m.consolidate_fact("c", "v", .7, [("a", "v")])
            m.consolidate_fact("d", "v", .7, [("b", "v"), ("c", "v")])
            m.consolidate_fact("e", "v", .7, [("d", "v")])
            m.save()
            m = self.memory(sharded, str(sharded))
            with mock.patch.object(m, "fact_keys", side_effect=AssertionError("whole scan")):
                self.assertEqual(set(m.derivatives_of("a")), {"b", "c", "d", "e"})
            saved = m.recall_fact("b")
            m.restore_fact("b", None)
            self.assertEqual(m._derived_candidates("a"), {"c"})
            m.restore_fact("b", saved)
            m.consolidate_fact("b", "v", .8, [("x", "v")])
            self.assertEqual(m._derived_candidates("a"), {"c"})
            m.remember_fact("c", "new")
            self.assertEqual(m._derived_candidates("a"), set())
            self.assertIsNone(m.recall_fact("e"))

    def test_candidates_are_verified_and_seam_is_the_only_source(self):
        m = self.memory(False)
        m.remember_fact("unrelated", "v")
        m.consolidate_fact("derived", "v", .8, [("a", "v")])
        with mock.patch.object(m, "_derived_candidates", return_value=set()):
            self.assertEqual(m.derivatives_of("a"), [])
        with mock.patch.object(m, "_derived_candidates", return_value={"unrelated", "derived", "gone"}):
            self.assertEqual(m.derivatives_of("a"), ["derived"])

    def test_old_derivations_are_indexed_once(self):
        vault = ShardVault(self.root / "memory_vault")
        fs = FactShards(vault)
        key = "derived key"
        bucket = fs.bucket_of(key)
        vault.add_shard(bucket, bucket, {"facts": {key: {
            "value": "v", "derived_from": [["premise", "v"]]}}}, kind="fact-bucket")
        m = self.memory()
        self.assertEqual(m.derivatives_of("premise"), [key])
        m.save()
        fresh = self.memory()
        # §11.139: even the first query after restart reads only index pages.
        with mock.patch.object(fresh.shards, "_load", side_effect=AssertionError("scan")):
            self.assertEqual(fresh._derived_candidates("premise"), {key})
        with mock.patch.object(fresh.shards, "keys", side_effect=AssertionError("scan")):
            self.assertEqual(fresh.derivatives_of("premise"), [key])
        loaded = []
        real = fresh.shards._load
        with mock.patch.object(fresh.shards, "_load", side_effect=lambda sid: (loaded.append(sid), real(sid))[1]):
            self.assertEqual(fresh.derivatives_of("unrelated"), [])
        self.assertEqual(loaded, [])


# §11.139: the writer, migration and reader use explicit structure only.
class StructureFlowTests(unittest.TestCase):
    setUp = CatalogueTests.setUp
    memory = CatalogueTests.memory

    def stash(self):
        from ultraquant.interpreter.stash import ContemporaryStash
        return ContemporaryStash(self.root / "stash.json")

    def claim(self, stash, subject="Kenya", fields=None, provenance=True):
        return stash.add_claim(
            f"https://distill.invalid/run/capital:{subject}", "Question",
            f"The capital of {subject} is Nairobi.",
            provenance={"run_id": "run"} if provenance else None, fields=fields)

    def test_writer_passes_subject_and_template_attribute(self):
        from ultraquant.distill import elicit as E, file as F
        from ultraquant.experiments import knowledge_bench as K
        stash = self.stash()
        items = [K.Item(category, subject, "Question", (value.lower(),))
                 for category, subject, value in [("capital", "Kenya", "Nairobi"),
                     ("symbol", "gold", "Au"), ("author", "a book", "Writer"),
                     ("number", "gold", "79")]]
        samples = [E.Record(teacher, teacher, "x", 1, E.question_id(item),
                            item.question, seed, value, value.lower())
                   for item, value in zip(items, ["Nairobi", "Au", "Writer", "79"])
                   for teacher in ("a", "b") for seed in E.SEEDS]
        ids = F.file_distilled(stash, samples, items, .9, "run")
        m = self.memory()
        for eid, item, attribute in zip(ids, items, ["capital", "chemical symbol", "author", "atomic number"]):
            fields = stash.get(eid)["fields"]
            self.assertEqual(fields["subject"], item.subject)
            self.assertEqual(fields["attribute"], attribute)
            with mock.patch.object(m, "remember_fact", wraps=m.remember_fact) as remember:
                key = stash.promote(eid, m)
                self.assertEqual(remember.call_args.kwargs["subject"], item.subject)
                self.assertEqual(remember.call_args.kwargs["attribute"], attribute)
            self.assertEqual(m.recall_fact(key)["subject"], item.subject)
        self.assertEqual(len(ids), 4)

    def test_structure_of_requires_provenance_and_exact_suffix(self):
        from ultraquant.memory.migrate import _structure_of
        stash = self.stash()
        eid = self.claim(stash, "The County: North")
        self.assertEqual(_structure_of(stash.get(eid)), ("The County: North", "capital"))
        entry = stash.get(eid)
        entry["provenance"] = None
        self.assertIsNone(_structure_of(entry))
        entry = stash.get(eid)
        entry["fields"] = {"key": "county capital", "value": "v"}
        self.assertIsNone(_structure_of(entry))
        entry["fields"]["key"] = "capital of the county: north extra"
        self.assertIsNone(_structure_of(entry))
        entry["fields"]["key"] = "symbol of the county: north"
        self.assertEqual(_structure_of(entry), ("The County: North", "symbol"))

    def test_migration_is_lossless_idempotent_and_provenance_only(self):
        from ultraquant.memory.migrate import structure_from_provenance
        stash, m = self.stash(), self.memory()
        first = self.claim(stash)
        second = self.claim(stash, "Ghana", fields={"key": "capital of ghana", "value": "Accra"})
        chat = self.claim(stash, "Peru", provenance=False)
        pending = self.claim(stash, "Chile")
        missing = self.claim(stash, "Missing")
        for eid in (first, second, chat):
            stash.promote(eid, m)
        # §11.139: completing a promoted claim must not resurrect a dropped fact.
        stash.promote(missing, m)
        m.restore_fact("capital of missing", None)
        before = {key: m.recall_fact(key) for key in m.fact_keys()}
        order = []
        save_stash, save_memory = stash.save, m.save
        with mock.patch.object(stash, "save", side_effect=lambda: (order.append("stash"), save_stash())[1]), \
                mock.patch.object(m, "save", side_effect=lambda: (order.append("memory"), save_memory())[1]):
            counts = structure_from_provenance(m, stash)
        self.assertEqual(order, ["stash", "memory"])
        self.assertEqual(counts["facts"], 2)
        self.assertEqual(counts["entries"], 3)
        for key, record in before.items():
            now = m.recall_fact(key)
            self.assertEqual({k: v for k, v in now.items() if k not in ("subject", "attribute")}, record)
        self.assertNotIn("subject", m.recall_fact("capital of peru"))
        self.assertIsNone(m.recall_fact("capital of chile"))
        self.assertIsNone(m.recall_fact("capital of missing"))
        self.assertEqual(stash.get(missing)["fields"]["subject"], "Missing")
        self.assertEqual(stash.get(pending)["status"], "staged")
        self.assertEqual(stash.get(first)["fields"], {"key": "capital of kenya", "value": "Nairobi",
                                                       "subject": "Kenya", "attribute": "capital"})
        self.assertEqual(structure_from_provenance(m, stash), {"facts": 0, "entries": 0})
        self.assertEqual(self.memory().recall_fact("capital of ghana")["subject"], "Ghana")

    def test_migration_calls_its_structure_seam(self):
        from ultraquant.memory import migrate
        stash, m = self.stash(), self.memory()
        eid = self.claim(stash)
        stash.promote(eid, m)
        with mock.patch.object(migrate, "_structure_of", return_value=None) as structure:
            self.assertEqual(migrate.structure_from_provenance(m, stash), {"facts": 0, "entries": 0})
            structure.assert_called_once()
        self.assertNotIn("subject", m.recall_fact("capital of kenya"))

    def test_approval_identity_ignores_only_catalogue_fields(self):
        from ultraquant.interpreter.autoapprove import AutoApprover
        from ultraquant.memory.migrate import structure_from_provenance
        stash, m = self.stash(), self.memory()
        self.claim(stash)
        approver = AutoApprover(stash, m, self.root / "approvals.jsonl")
        approval, = approver.approve_all()
        structure_from_provenance(m, stash)
        self.assertEqual(approver._dispute_mode(approval), "exact")
        self.assertTrue(approver._still_holds(approval.key, approval.after))
        legacy = copy.deepcopy(approval)
        legacy.after = None
        self.assertEqual(approver._dispute_mode(legacy), "exact")
        record = m.recall_fact(approval.key)
        record["confidence"] += .01
        m.restore_fact(approval.key, record)
        self.assertFalse(approver._still_holds(approval.key, approval.after))
        self.assertEqual(approver._dispute_mode(approval), "superseded")
        record["confidence"] = approval.after["confidence"]
        m.restore_fact(approval.key, record)
        approver.dispute(approval.key, "test")
        self.assertIsNone(m.recall_fact(approval.key))

    def test_curiosity_checks_catalogue_once_and_honors_seam(self):
        from ultraquant.reason import inference as I
        self.assertTrue(I._bridge_allowed({}, set()))
        self.assertTrue(I._bridge_allowed({"subject": "The TówER"}, {"tower"}))
        self.assertFalse(I._bridge_allowed({"subject": "tower"}, {"spire"}))
        m = self.memory(False)
        m.remember_fact("material of the tower", "steel", subject="tower", attribute="material")
        with mock.patch.object(m, "subjects_in", wraps=m.subjects_in) as lookup:
            gap = I.missing_premise("What is the conductivity of the tower?", m)
            self.assertEqual(gap["premise_key"], "steel conductivity")
            lookup.assert_called_once()
        with mock.patch.object(I, "_bridge_allowed", return_value=False) as bridge:
            self.assertIsNone(I.missing_premise("What is the conductivity of the tower?", m))
            self.assertTrue(bridge.called)
        m = self.memory(False, "chat")
        m.remember_fact("tower material", "steel")
        self.assertEqual(I.missing_premise("What is the tower conductivity?", m)["premise_key"],
                         "steel conductivity")


if __name__ == "__main__":
    unittest.main()
