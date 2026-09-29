"""Teacher identities come from committed model-file evidence."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.distill import elicit, file, frontier, sources, targets
from ultraquant.interpreter.stash import ContemporaryStash


class TeacherIdentityTests(unittest.TestCase):
    def test_index_matches_committed_records_and_later_specs(self):
        root = Path(__file__).resolve().parents[1]

        def git(*args):
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, encoding="utf-8")

        evidence = {}
        for path in git("ls-tree", "-r", "--name-only", "HEAD",
                        "ultraquant/experiments/records").splitlines():
            if not path.endswith(".jsonl"):
                continue
            for line in git("show", f"HEAD:{path}").splitlines():
                row = json.loads(line)
                if row.get("teacher") and row.get("gguf_name") and row.get("gguf_size"):
                    identity = f"{row['gguf_name']}:{row['gguf_size']}"
                    self.assertEqual(evidence.setdefault(row["teacher"], identity), identity)
        self.assertTrue(evidence)
        evidence.update({
            "qwen/qwen3.8-27b": "Qwen3.8-27B-Q4_K_M.gguf:16810714336",
            "cydonia-v1.3-magnum-v4-22b": "Cydonia-v1.3-Magnum-v4-22B-Q6_K.gguf:18252706816",
        })
        self.assertEqual(sources._teacher_index(), evidence)

    def test_three_alias_pairs_share_three_distinct_identities(self):
        pairs = [("command-r-08-2024", "c4ai-command-r-08-2024"),
                 ("qwen3.8-27b", "qwen/qwen3.8-27b"),
                 ("cydonia-22b", "cydonia-v1.3-magnum-v4-22b")]
        for old, new in pairs:
            with self.subTest(old=old, new=new):
                self.assertEqual(sources.identity(old), sources.identity(new))
        self.assertEqual(len({sources.identity(old) for old, _ in pairs}), 3)

    def test_unknown_name_and_module_index_seam(self):
        self.assertEqual(sources.identity("unlisted teacher"), "unlisted teacher")
        with mock.patch.object(sources, "_teacher_index", return_value={}):
            self.assertEqual(sources.identity("qwen3.8-27b"), "qwen3.8-27b")

    def test_filers_use_supporting_record_metadata_in_teacher_order(self):
        forward = targets.Target("number", "Boron", "Forward?", "atomic number")
        reverse = frontier.ReverseTarget("number", "5", "Reverse?", "atomic number", "5")
        for item, raw, filer in ((forward, "5", file.file_distilled),
                                 (reverse, "Boron", frontier.file_reverse)):
            with self.subTest(filer=filer.__name__), tempfile.TemporaryDirectory() as home:
                stash = ContemporaryStash(Path(home) / "stash.json")
                records = [elicit.Record(name, lineage, gguf, size,
                                         elicit.question_id(item), item.question, seed,
                                         answer, elicit.normalize(answer))
                           for name, lineage, gguf, size, answer in (
                               ("qwen3.8-27b", "Qwen", "recorded-qwen.gguf", 42, raw),
                               ("command-r-08-2024", "Command", "recorded-command.gguf", 84, raw),
                               ("unsupported", "Other", "other.gguf", 99, "UNKNOWN"))
                           for seed in elicit.SEEDS]
                entry_id, = filer(stash, records, [item], 0.9, "metadata")
                stored = stash.get(entry_id)["provenance"]
                self.assertEqual(stored["teachers"], ["command-r-08-2024", "qwen3.8-27b"])
                self.assertEqual(stored["teacher_ids"],
                                 ["recorded-command.gguf:84", "recorded-qwen.gguf:42"])
                self.assertEqual(stored["lineages"], ["Command", "Qwen"])


if __name__ == "__main__":
    unittest.main()
