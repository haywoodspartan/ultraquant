"""Paid GPUs attach on demand.

11.127 made the runner that is the only way this project may touch a
rented GPU; 11.128 rebuilt it after GPT-6 Astra's adversarial review
found 13 defects in it. These pins hold the parsing against the real
lupine v0.3.1 text captured on 2026-09-28, the ledger's accounting,
the argv the supervisor is launched with, the paths the review
reproduced (a process tree outliving a kill, an interrupt mid-job, a
supervisor that dies at once), and the exam's own exit status. The
whole exam spends nothing - it runs against a stand-in CLI - but takes
a minute or two: set ULTRAQUANT_SLOW_GATES=1 to run it here.
"""

from __future__ import annotations

import calendar
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.cloud import ondemand as od
from ultraquant.cloud import supervisor as sup
from ultraquant.experiments import lupine_stub as S
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
REAL_LEASE = ("lupine: lease b5f8f284-32b9-4829-a39d-125821bac1af on "
              "lupine-server-597f59457c-jmh6p (NVIDIA RTX PRO 6000 "
              "Blackwell Server Edition)")


class ParsingTests(unittest.TestCase):
    """What the real CLI printed, read the way the runner reads it."""

    def test_the_price_list(self) -> None:
        rates = od.parse_gpus(REAL_GPUS)
        self.assertEqual(rates.by_sku, {"a100": 1.10, "rtx-pro-6000": 1.50})
        self.assertEqual(rates.for_sku(None), 1.50)   # unpinned: the dearest
        with self.assertRaises(ValueError):
            rates.for_sku("h100")

    def test_a_price_list_is_read_whole_or_refused(self) -> None:
        """Review F7: an unreadable row once vanished and made the
        cheapest GPU the dearest one."""
        for broken in (REAL_GPUS.replace("$1.50", "$1.50/hr"),
                       REAL_GPUS + "A new version of lupine is available\n",
                       "SKU  GPU  VRAM  $/HR\n",
                       "lupine: error: not logged in\n"):
            with self.subTest(broken=broken[-40:]):
                with self.assertRaises(ValueError):
                    od.parse_gpus(broken)

    def test_the_usage_summary(self) -> None:
        self.assertEqual(od.parse_usage(REAL_USAGE),
                         {"gpu_seconds": 1, "cost": 0.0})
        for shown, seconds in (("2m 5s", 125), ("1h 2m 3s", 3723),
                               ("0s", 0)):
            with self.subTest(shown=shown):
                self.assertEqual(
                    od.parse_usage(f"GPU time: {shown}\nEstimated cost: $1.25\n"),
                    {"gpu_seconds": seconds, "cost": 1.25})
        self.assertIsNone(od.parse_usage("GPU time: 1d 2h\nEstimated cost: $1\n"))

    def test_the_lease_line(self) -> None:
        self.assertEqual(sup.parse_lease("noise\n" + REAL_LEASE + "\n"),
                         ("b5f8f284-32b9-4829-a39d-125821bac1af",
                          "NVIDIA RTX PRO 6000 Blackwell Server Edition"))
        self.assertEqual(sup.parse_lease("lupine: error\n"), (None, None))


class LaunchTests(unittest.TestCase):
    """How lupine and its supervisor are started, and with what."""

    def test_windows_paths_become_wsl_paths(self) -> None:
        self.assertEqual(od.to_wsl_path(r"H:\AI Model AGI\x.py"),
                         "/mnt/h/AI Model AGI/x.py")
        self.assertEqual(od.to_wsl_path("C:/a/b"), "/mnt/c/a/b")
        with self.assertRaises(ValueError):
            od.to_wsl_path("relative/path")

    def test_the_supervised_command_is_exact(self) -> None:
        launcher = od.Launcher(prefix=("/home/u/.local/bin/lupine",),
                               wsl_distro="Ubuntu",
                               supervisor=r"H:\repo\sup.py")
        self.assertEqual(
            launcher.supervised(["run", "--", "x"], workdir=r"H:\w",
                                deadline=60.0, kill_grace=10.0,
                                release_attempts=3, release_timeout=5.0),
            ["wsl.exe", "-d", "Ubuntu", "--cd", r"H:\w", "--", "python3",
             "/mnt/h/repo/sup.py", "--deadline", "60", "--kill-grace", "10",
             "--release-attempts", "3", "--release-timeout", "5",
             "--lupine", "/home/u/.local/bin/lupine", "--", "run", "--",
             "x"])
        self.assertIsNone(launcher.cwd(r"H:\w"))

    def test_the_child_environment_hides_the_local_gpu(self) -> None:
        env = sup.child_environment({
            "PATH": os.pathsep.join(["/usr/bin", "/mnt/c/Windows/System32",
                                     "/mnt/c/Python313"]),
            "LUPINE_DISABLE_LOCAL": "0"})
        self.assertEqual(env["LUPINE_DISABLE_LOCAL"], "1")
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_wslenv_replaces_a_stale_entry(self) -> None:
        runner = od.OnDemand(od.Launcher(prefix=("lupine",)),
                             od.Ledger(Path(tempfile.gettempdir()) / "u.jsonl"),
                             1.0, lock_path=Path(tempfile.gettempdir())
                             / "u.lock")
        with mock.patch.dict(os.environ,
                             {"WSLENV": "A/p::LUPINE_DISABLE_LOCAL/p:B",
                              "LUPINE_DISABLE_LOCAL": "no"}):
            env = runner._environment()
        self.assertEqual(env["WSLENV"], "A/p:B:LUPINE_DISABLE_LOCAL/u")
        self.assertEqual(env["LUPINE_DISABLE_LOCAL"], "1")


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

    def test_windows_programs_are_refused_unless_asked_for(self) -> None:
        """Review F6: WSL runs Windows programs outside lupine's shim, where
        the local RTX 4090 is visible while the rental bills."""
        for command in (("/mnt/c/Python313/python.exe", "x.py"),
                        ("nvidia-smi.EXE",), ("/mnt/d/tool",)):
            with self.subTest(command=command):
                with self.assertRaises(ValueError):
                    od.Job(command=command, max_seconds=1.0, label="x")
        self.assertTrue(od.Job(command=("/mnt/d/tool",), max_seconds=1.0,
                               label="x", allow_interop=True).allow_interop)

    def test_a_cap_must_be_a_real_amount(self) -> None:
        for cap in (-1.0, float("inf"), float("nan")):
            with self.subTest(cap=cap):
                with self.assertRaises(ValueError):
                    od.OnDemand(od.Launcher(prefix=("lupine",)),
                                od.Ledger(Path(tempfile.gettempdir())
                                          / "unused.jsonl"), cap)

    def test_grace_covers_the_worst_case_it_is_built_from(self) -> None:
        runner = od.OnDemand(od.Launcher(prefix=("lupine",)),
                             od.Ledger(Path(tempfile.gettempdir()) / "g.jsonl"),
                             1.0, kill_grace=10.0, release_attempts=3,
                             release_timeout=5.0, drain_timeout=5.0)
        # kill 10 + two bounded drains + two releases of 3 x 5 s + backoff
        self.assertGreaterEqual(runner.grace(), 10 + 2 * 5 + 2 * (15 + 1.5))


class LedgerTests(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_ledger_"))
        self.ledger = od.Ledger(self.dir / "ledger.jsonl")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _receipt(self, started: float, cost: float, *, released=True,
                 attached=1.0, reservation="") -> od.Receipt:
        return od.Receipt(label="t", started=started, seconds=1.0, sku=None,
                          rate=1.5, cost=cost, worst_case=cost, exit_code=0,
                          timed_out=False, lease=None, gpu=None,
                          released=released, release_output="",
                          output_tail="", attached_seconds=attached,
                          reservation=reservation)

    def test_the_month_is_utc_and_a_spanning_job_counts_in_both(self) -> None:
        last = calendar.timegm((2026, 9, 30, 23, 59, 50))
        self.ledger.append(self._receipt(last - 3600, 2.0))
        self.ledger.append(self._receipt(last, 3.0, attached=20.0))
        october = calendar.timegm((2026, 10, 1, 12, 0, 0))
        self.assertEqual(self.ledger.spent_in_month(last), 5.0)
        self.assertEqual(self.ledger.spent_in_month(october), 3.0)

    def test_an_open_reservation_counts_until_its_receipt(self) -> None:
        now = time.time()
        self.ledger.reserve("r1", now, 0.30, "crashed")
        self.ledger.reserve("r2", now, 0.40, "finished")
        self.ledger.append(self._receipt(now, 0.01, reservation="r2"))
        self.assertAlmostEqual(self.ledger.spent_in_month(), 0.31)
        self.assertEqual(len(self.ledger.receipts()), 1)

    def test_an_unreleased_lease_stays_outstanding_until_cleared(self) -> None:
        now = time.time()
        self.assertIsNone(self.ledger.outstanding())
        self.ledger.append(self._receipt(now, 0.1, released=False))
        self.assertIsNotNone(self.ledger.outstanding())
        self.ledger.reconcile("ended on the next admission")
        self.assertIsNone(self.ledger.outstanding())
        self.ledger.append(self._receipt(now, 0.1, released=False))
        self.ledger.append(self._receipt(now, 0.1, released=True))
        self.assertIsNone(self.ledger.outstanding())

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


class _StubCase(unittest.TestCase):
    """A fresh stand-in lupine per test; nothing real is ever called."""

    faults: dict = {}

    def setUp(self) -> None:
        self.stub = G.Stub()
        if self.faults:
            self.stub.set(**self.faults)
        patcher = mock.patch.dict(os.environ,
                                  {"LUPINE_STUB_DIR": str(self.stub.dir)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.stub.cleanup)
        self.runner = G._runner(self.stub)


class JobPathTests(_StubCase):

    def test_a_clean_job_is_read_and_charged_from_launch_to_release(self) -> None:
        receipt = self.runner.run(G._job(["nvidia-smi", "-L"]))
        self.assertTrue(receipt.released)
        self.assertEqual(receipt.lease, self.stub.runs()[0]["lease"])
        self.assertEqual(receipt.gpu, S.GPU_NAME)
        self.assertIn("(via lupine", receipt.output_tail)
        self.assertGreaterEqual(receipt.attached_seconds, receipt.seconds)
        self.assertAlmostEqual(receipt.cost,
                               receipt.rate * receipt.attached_seconds / 3600)
        self.assertEqual(receipt.error, "")

    def test_a_process_tree_does_not_outlive_its_deadline(self) -> None:
        """Review F1: a grandchild holding stdout kept a 0.5 s job busy 6.2 s."""
        started = time.monotonic()
        receipt = self.runner.run(G._job(["sleep-tree", "30"], 1.0))
        self.assertLess(time.monotonic() - started, 1.0 + self.runner.grace())
        self.assertTrue(receipt.timed_out and receipt.released)
        time.sleep(0.5)
        self.assertEqual(self.stub.survivors(), [])
        self.assertIsNone(self.stub.lease())

    def test_an_interrupt_mid_job_still_releases(self) -> None:
        def interrupted(runner, process, timeout):
            time.sleep(1.5)
            raise KeyboardInterrupt

        with mock.patch.object(od.OnDemand, "_wait", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.run(G._job(["sleep", "30"]))
        time.sleep(0.5)
        self.assertIsNone(self.stub.lease())
        (row,) = self.stub.receipt_rows()
        self.assertTrue(row["released"] and row["interrupted"])
        self.assertFalse(row["timed_out"])
        self.assertEqual(self.stub.survivors(), [])

    def test_a_supervisor_that_dies_at_once_still_leaves_a_receipt(self) -> None:
        launcher = od.Launcher(prefix=self.runner.launcher.prefix,
                               supervisor=str(self.stub.dir / "missing.py"))
        runner = od.OnDemand(launcher, od.Ledger(self.stub.ledger_path), 100.0,
                             lock_path=self.stub.lock_path, **G.TIMING)
        try:
            receipt = runner.run(G._job(["echo", "x"]))
        except od.LeaseNotReleased as exc:      # also acceptable, if loud
            receipt = exc.receipt
        self.assertEqual(len(self.stub.receipt_rows()), 1)
        self.assertEqual(self.stub.runs(), [])  # nothing was ever leased
        self.assertIsNone(receipt.exit_code)

    def test_an_unknown_sku_never_launches(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.run(G._job(["echo", "x"], sku="h100"))
        self.assertEqual(self.stub.runs(), [])
        self.assertEqual(self.stub.receipt_rows(), [])


class RoundedUsageTests(_StubCase):
    """Review F8: "$0.00" can hide up to half a cent of real spend."""

    faults = {"usage_seconds": 11}

    def test_the_usage_figure_is_bounded_above(self) -> None:
        self.assertAlmostEqual(self.runner.month_spent(), 0.005)


class GateTests(unittest.TestCase):

    def test_the_whole_exam(self) -> None:
        if os.environ.get("ULTRAQUANT_SLOW_GATES") != "1":
            self.skipTest("set ULTRAQUANT_SLOW_GATES=1 for the on-demand exam")
        report = G.run_gate()
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(len(report.planted), 9)

    def test_the_exam_exits_nonzero_unless_it_passes(self) -> None:
        """Review F13: the exam used to exit 0 on a failing verdict."""
        for passes, code in ((True, 0), (False, 1)):
            report = G.OnDemandReport(passes=passes, reason="stubbed")
            with mock.patch.object(G, "run_gate", return_value=report), \
                    mock.patch("builtins.print"):
                self.assertEqual(G.main(), code)

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Release on every path", "Refused before attach",
                       "The local GPU stays hidden", "One lease at a time",
                       "The exam can fail", "was not ready"):
            self.assertIn(phrase, doc)
        self.assertIn("PASSED", doc)
        self.assertIn("9 of 9 planted defects", doc)


if __name__ == "__main__":
    unittest.main()
