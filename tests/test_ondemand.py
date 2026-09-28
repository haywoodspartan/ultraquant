"""Paid GPUs attach on demand.

11.127: the only way this project may touch a rented GPU. These pins
hold the parsing against the real lupine v0.3.1 text captured on
2026-09-28, the paths the exam did not cover (an interrupt mid-job, a
launch that fails, an unknown SKU), the ledger's month boundary, and
the whole exam - which runs against a stand-in CLI and spends nothing.
"""

from __future__ import annotations

import calendar
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.cloud import ondemand as od
from ultraquant.experiments import ondemand_gate as G

REAL_GPUS = (
    "SKU              GPU                                            VRAM     $/HR\n"
    "a100             NVIDIA A100-SXM4-80GB                          80GB    $1.10\n"
    "rtx-pro-6000     NVIDIA RTX PRO 6000 Blackwell Server Edition   96GB    $1.50\n"
    "\n"
    "Run on a type:  lupine run --gpu <SKU> <command>\n")
REAL_USAGE = ("Billing period:  Sep 2026 (month to date)\n"
              "Payment method:  on file\n"
              "GPU time:        1s\n"
              "Sessions:        1\n"
              "Estimated cost:  $0.00\n"
              "Free credit:     $20.00 remaining\n")


class ParsingTests(unittest.TestCase):
    """What the real CLI printed, read the way the runner reads it."""

    def test_the_price_list(self) -> None:
        rates = od.parse_gpus(REAL_GPUS)
        self.assertEqual(rates.by_sku, {"a100": 1.10, "rtx-pro-6000": 1.50})
        self.assertEqual(rates.for_sku("a100"), 1.10)
        self.assertEqual(rates.for_sku(None), 1.50)   # unpinned: the dearest
        with self.assertRaises(ValueError):
            rates.for_sku("h100")
        with self.assertRaises(ValueError):
            od.parse_gpus("lupine: error: not logged in\n")

    def test_the_usage_summary(self) -> None:
        self.assertEqual(od.parse_usage(REAL_USAGE),
                         {"gpu_seconds": 1, "cost": 0.0})
        for shown, seconds in (("2m 5s", 125), ("1h 2m 3s", 3723),
                               ("3h", 10800), ("0s", 0)):
            with self.subTest(shown=shown):
                parsed = od.parse_usage(f"GPU time: {shown}\n"
                                        "Estimated cost: $1.25\n")
                self.assertEqual(parsed, {"gpu_seconds": seconds,
                                          "cost": 1.25})
        # anything unreadable is None, and None means refusing to run
        for broken in ("GPU time: \nEstimated cost: $1.25\n",
                       "GPU time: 1d 2h\nEstimated cost: $1.25\n",
                       "GPU time: 5s\n"):
            with self.subTest(broken=broken):
                self.assertIsNone(od.parse_usage(broken))


class _StubCase(unittest.TestCase):
    """A fresh stand-in lupine per test; nothing real is ever called."""

    def setUp(self) -> None:
        self.stub = G.Stub()
        patcher = mock.patch.dict(os.environ,
                                  {"LUPINE_STUB_DIR": str(self.stub.dir)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.stub.cleanup)
        self.runner = G._runner(self.stub)

    def receipts(self) -> list:
        return od.Ledger(self.stub.ledger_path).receipts()


class PathsTheExamDidNotCoverTests(_StubCase):

    def test_the_lease_line_is_read(self) -> None:
        from ultraquant.experiments import lupine_stub as S
        receipt = self.runner.run(G._job(["nvidia-smi", "-L"]))
        self.assertEqual(receipt.lease, self.stub.runs()[0]["lease"])
        self.assertEqual(receipt.gpu, S.GPU_NAME)
        self.assertIn("(via lupine", receipt.output_tail)

    def test_an_interrupt_mid_job_still_releases(self) -> None:
        """Review finding: an interrupted job was recorded as timed out."""
        real = subprocess.Popen.communicate
        state = {"hit": False}

        def interrupting(process, *args, **kwargs):
            if ("run" in process.args and not state["hit"]
                    and kwargs.get("timeout")):
                state["hit"] = True
                time.sleep(0.8)         # let the stub start and cache a lease
                raise KeyboardInterrupt
            return real(process, *args, **kwargs)

        with mock.patch.object(subprocess.Popen, "communicate", interrupting):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.run(G._job(["sleep", "5"]))
        self.assertEqual(len(self.stub.runs()), 1)
        self.assertIsNone(self.stub.lease())
        (receipt,) = self.receipts()
        self.assertTrue(receipt["released"])
        self.assertFalse(receipt["timed_out"])
        self.assertTrue(receipt["error"].startswith("KeyboardInterrupt"))

    def test_a_launch_that_fails_still_releases_and_records(self) -> None:
        original = od.OnDemand._launch

        def failing(runner, args, **kwargs):
            if args and args[0] == "run":
                raise FileNotFoundError("planted: no launcher")
            return original(runner, args, **kwargs)

        with mock.patch.object(od.OnDemand, "_launch", failing):
            with self.assertRaises(FileNotFoundError):
                self.runner.run(G._job(["echo", "x"]))
        ends = [e for e in self.stub.log() if e.get("cmd") == "end"]
        self.assertEqual(len(ends), 1)
        (receipt,) = self.receipts()
        self.assertTrue(receipt["released"])
        self.assertIsNone(receipt["exit_code"])
        self.assertIn("FileNotFoundError", receipt["error"])

    def test_an_unknown_sku_never_launches(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.run(G._job(["echo", "x"], sku="h100"))
        self.assertEqual(self.stub.runs(), [])
        self.assertEqual(self.receipts(), [])

    def test_a_clean_job_records_no_error(self) -> None:
        receipt = self.runner.run(G._job(["exit", "3"]))
        self.assertEqual(receipt.exit_code, 3)   # a failing workload is data
        self.assertFalse(receipt.timed_out)
        self.assertEqual(receipt.error, "")


class LedgerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_ledger_"))
        self.ledger = od.Ledger(self.dir / "ledger.jsonl")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _receipt(self, started: float, cost: float) -> od.Receipt:
        return od.Receipt(label="t", started=started, seconds=1.0, sku=None,
                          rate=1.5, cost=cost, worst_case=cost, exit_code=0,
                          timed_out=False, lease=None, gpu=None,
                          released=True, release_output="", output_tail="")

    def test_the_month_is_utc(self) -> None:
        last_second = calendar.timegm((2026, 9, 30, 23, 59, 59))
        self.ledger.append(self._receipt(last_second, 2.0))
        self.ledger.append(self._receipt(last_second + 1, 3.0))
        self.assertEqual(self.ledger.spent_in_month(last_second), 2.0)
        self.assertEqual(self.ledger.spent_in_month(last_second + 1), 3.0)

    def test_a_corrupt_line_is_skipped_not_fatal(self) -> None:
        self.ledger.append(self._receipt(time.time(), 0.25))
        with self.ledger.path.open("a", encoding="utf-8") as handle:
            handle.write('{"label": "torn wri\n')
        self.ledger.append(self._receipt(time.time(), 0.5))
        self.assertEqual(len(self.ledger.receipts()), 2)
        self.assertAlmostEqual(self.ledger.spent_in_month(), 0.75)

    def test_the_default_location_honours_the_override(self) -> None:
        target = str(self.dir / "elsewhere.jsonl")
        with mock.patch.dict(os.environ, {"ULTRAQUANT_GPU_LEDGER": target}):
            self.assertEqual(str(od.Ledger.default().path), target)

    def test_receipts_round_trip_as_json(self) -> None:
        self.ledger.append(self._receipt(1.0, 0.1))
        row = json.loads(self.ledger.path.read_text(encoding="utf-8"))
        self.assertEqual(row["error"], "")
        self.assertEqual(row["cost"], 0.1)


class ContractTests(unittest.TestCase):

    def test_a_job_must_say_what_it_is_and_when_it_stops(self) -> None:
        for bad in ({"command": (), "max_seconds": 1.0, "label": "x"},
                    {"command": ("x",), "max_seconds": 0.0, "label": "x"},
                    {"command": ("x",), "max_seconds": float("inf"),
                     "label": "x"},
                    {"command": ("x",), "max_seconds": float("nan"),
                     "label": "x"},
                    {"command": ("x",), "max_seconds": 1.0, "label": ""}):
            with self.subTest(job=bad):
                with self.assertRaises(ValueError):
                    od.Job(**bad)

    def test_a_cap_must_be_a_real_amount(self) -> None:
        launcher = od.Launcher(prefix=("lupine",))
        ledger = od.Ledger(Path(tempfile.gettempdir()) / "unused.jsonl")
        for cap in (-1.0, float("inf"), float("nan")):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    od.OnDemand(launcher, ledger, cap)

    def test_wslenv_replaces_a_stale_entry(self) -> None:
        runner = od.OnDemand(od.Launcher(prefix=("lupine",)),
                             od.Ledger(Path(tempfile.gettempdir()) / "u.jsonl"),
                             1.0)
        with mock.patch.dict(os.environ,
                             {"WSLENV": "A/p::LUPINE_DISABLE_LOCAL/p:B",
                              "LUPINE_DISABLE_LOCAL": "no"}):
            env = runner._environment()
        self.assertEqual(env["WSLENV"], "A/p:B:LUPINE_DISABLE_LOCAL/u")
        self.assertEqual(env["LUPINE_DISABLE_LOCAL"], "1")


class GateTests(unittest.TestCase):

    def test_the_whole_exam(self) -> None:
        """About 12 s: every criterion and every planted defect."""
        report = G.run_gate()
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(len(report.planted), 5)

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Release on every path", "Refused before attach",
                       "The local GPU stays hidden", "One lease at a time",
                       "The exam can fail"):
            self.assertIn(phrase, doc)
        self.assertIn("PASSED", doc)
        self.assertIn("5 of 5 planted defects", doc)


if __name__ == "__main__":
    unittest.main()
