"""Tests for the BlueQubit cloud backend, driven through a stub SDK.

The real platform needs an account, a token, and credit for the GPU and QPU
devices, so what is verified here is everything on our side of the boundary:
availability gating, circuit translation, the exact kwargs sent to ``run()``,
both result paths (statevector and counts), bit ordering, and failure handling.
What is *not* verified is that the live service behaves as its documentation
says.
"""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest import mock

from ultraquant.quantum.backend import BackendUnavailable
from ultraquant.quantum.bluequbit import BLUEQUBIT_DEVICES, BlueQubitBackend
from ultraquant.quantum.circuit import Circuit


class _StubResult:
    """Stands in for a BlueQubit JobResult."""

    def __init__(self, counts=None, statevector=None, error_message=None) -> None:
        self._counts = counts
        self._statevector = statevector
        self.error_message = error_message
        self.job_id = "job-123"
        self.device = "cpu"
        self.run_status = "COMPLETED"

    def get_counts(self):
        if self._counts is None:
            raise RuntimeError("no counts for this job")
        return self._counts

    def get_statevector(self):
        if self._statevector is None:
            raise RuntimeError("no statevector for this job")
        return self._statevector


class _StubClient:
    """Records what the backend asked for, and returns a canned result."""

    def __init__(self, api_token=None, result=None, raises=None) -> None:
        self.api_token = api_token
        self.calls: list[tuple] = []
        self._result = result
        self._raises = raises

    def run(self, circuit, **kwargs):
        self.calls.append((circuit, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._result

    def estimate(self, circuit, device="cpu"):
        result = types.SimpleNamespace(estimated_runtime=1.5, estimated_cost=0.02)
        return result


def _install_stub(result=None, raises=None) -> _StubClient:
    """Put a fake ``bluequbit`` module in sys.modules and return its client."""
    client = _StubClient(result=result, raises=raises)
    module = types.ModuleType("bluequbit")
    module.BQClient = lambda api_token=None, **kw: (
        setattr(client, "api_token", api_token) or client
    )
    sys.modules["bluequbit"] = module
    return client


def _remove_stub() -> None:
    sys.modules.pop("bluequbit", None)


class AvailabilityTests(unittest.TestCase):
    """The tier gates on both the SDK and a token, and never raises."""

    def tearDown(self) -> None:
        _remove_stub()

    def test_absent_without_a_token(self) -> None:
        _install_stub()
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(BlueQubitBackend.available())

    def test_absent_without_the_sdk(self) -> None:
        _remove_stub()
        with mock.patch.dict("os.environ", {"BLUEQUBIT_API_TOKEN": "t"}):
            with mock.patch.dict(sys.modules, {"bluequbit": None}):
                self.assertFalse(BlueQubitBackend.available())

    def test_available_with_both(self) -> None:
        _install_stub()
        with mock.patch.dict("os.environ", {"BLUEQUBIT_API_TOKEN": "t"}):
            self.assertTrue(BlueQubitBackend.available())

    def test_missing_token_raises_unavailable(self) -> None:
        _install_stub()
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(BackendUnavailable):
                BlueQubitBackend()

    def test_unknown_device_rejected(self) -> None:
        _install_stub()
        with self.assertRaises(ValueError):
            BlueQubitBackend(device="quantum-teapot", api_token="t")

    def test_every_documented_device_constructs(self) -> None:
        _install_stub()
        for device in BLUEQUBIT_DEVICES:
            with self.subTest(device=device):
                backend = BlueQubitBackend(device=device, api_token="t")
                self.assertEqual(backend.name, f"bluequbit-{device}")


class ExpectationMathTests(unittest.TestCase):
    """Bit ordering and expectation maths match the rest of the project."""

    def test_counts_use_qubit_zero_rightmost(self) -> None:
        # "01" means qubit 0 = 1, qubit 1 = 0.
        got = BlueQubitBackend._expectations_from_counts({"01": 100}, 2)
        self.assertAlmostEqual(got[0], -1.0)
        self.assertAlmostEqual(got[1], +1.0)

    def test_bell_counts_average_to_zero(self) -> None:
        got = BlueQubitBackend._expectations_from_counts({"00": 500, "11": 500}, 2)
        self.assertAlmostEqual(got[0], 0.0)
        self.assertAlmostEqual(got[1], 0.0)

    def test_statevector_matches_the_local_simulator(self) -> None:
        root = 1 / math.sqrt(2)
        bell = [complex(root), 0j, 0j, complex(root)]
        got = BlueQubitBackend._expectations_from_statevector(bell, 2)
        self.assertAlmostEqual(got[0], 0.0, places=12)
        self.assertAlmostEqual(got[1], 0.0, places=12)

    def test_statevector_little_endian(self) -> None:
        # index 1 = qubit 0 set.
        got = BlueQubitBackend._expectations_from_statevector([0j, 1 + 0j], 1)
        self.assertAlmostEqual(got[0], -1.0)


class RunTests(unittest.TestCase):
    """Submission kwargs and both result paths."""

    def setUp(self) -> None:
        self.circuit = Circuit(2).h(0).cnot(0, 1)

    def tearDown(self) -> None:
        _remove_stub()

    def _backend(self, result, device="cpu", **kwargs):
        client = _install_stub(result=result)
        backend = BlueQubitBackend(device=device, api_token="t", **kwargs)
        return backend, client

    def test_exact_run_uses_the_statevector(self) -> None:
        root = 1 / math.sqrt(2)
        result = _StubResult(statevector=[complex(root), 0j, 0j, complex(root)])
        backend, client = self._backend(result)
        got = backend.run(self.circuit, shots=None)
        self.assertIsNone(got.counts)
        for value in got.expectations_z:
            self.assertAlmostEqual(value, 0.0, places=12)
        _circuit, kwargs = client.calls[0]
        self.assertNotIn("shots", kwargs, "an exact run must not request shots")
        self.assertEqual(kwargs["device"], "cpu")

    def test_sampled_run_requests_shots_and_returns_counts(self) -> None:
        result = _StubResult(counts={"00": 512, "11": 512})
        backend, client = self._backend(result)
        got = backend.run(self.circuit, shots=1024)
        self.assertEqual(got.counts, {"00": 512, "11": 512})
        self.assertAlmostEqual(got.expectations_z[0], 0.0)
        _circuit, kwargs = client.calls[0]
        self.assertEqual(kwargs["shots"], 1024)

    def test_quantum_device_always_samples(self) -> None:
        """Real hardware cannot hand back a statevector."""
        result = _StubResult(counts={"00": 600, "11": 424})
        backend, client = self._backend(result, device="quantum")
        backend.run(self.circuit, shots=None)
        _circuit, kwargs = client.calls[0]
        self.assertIn("shots", kwargs)
        self.assertEqual(kwargs["device"], "quantum")

    def test_statevector_failure_falls_back_to_counts(self) -> None:
        result = _StubResult(counts={"00": 10, "11": 10}, statevector=None)
        backend, _client = self._backend(result)
        got = backend.run(self.circuit, shots=None)
        self.assertIsNotNone(got.counts)

    def test_options_and_job_name_are_passed_through(self) -> None:
        result = _StubResult(counts={"00": 1})
        backend, client = self._backend(
            result, device="mps.cpu", job_name="lib-build",
            options={"mps_bond_dimension": 2},
        )
        backend.run(self.circuit, shots=8)
        _circuit, kwargs = client.calls[0]
        self.assertEqual(kwargs["job_name"], "lib-build")
        self.assertEqual(kwargs["options"], {"mps_bond_dimension": 2})

    def test_job_error_becomes_backend_unavailable(self) -> None:
        backend, _client = self._backend(_StubResult(error_message="out of credit"))
        with self.assertRaises(BackendUnavailable) as caught:
            backend.run(self.circuit, shots=16)
        self.assertIn("out of credit", str(caught.exception))

    def test_transport_failure_becomes_backend_unavailable(self) -> None:
        client = _install_stub(raises=RuntimeError("connection reset"))
        backend = BlueQubitBackend(api_token="t")
        with self.assertRaises(BackendUnavailable):
            backend.run(self.circuit, shots=16)
        self.assertTrue(client.calls)

    def test_empty_counts_is_an_error_not_a_silent_zero(self) -> None:
        backend, _client = self._backend(_StubResult(counts={}))
        with self.assertRaises(BackendUnavailable):
            backend.run(self.circuit, shots=16)

    def test_wide_circuit_never_asks_for_a_statevector(self) -> None:
        """A 40-qubit statevector is 17 TB; exactness must yield to sanity."""
        result = _StubResult(counts={"0" * 30: 512, "1" * 30: 512})
        backend, client = self._backend(result)
        backend.run(Circuit(30).h(0), shots=None)
        _circuit, kwargs = client.calls[0]
        self.assertIn("shots", kwargs, "a wide circuit must be sampled, not solved")
        self.assertTrue(any("statevector limit" in w for w in backend.last_warnings))

    def test_narrow_circuit_still_solved_exactly(self) -> None:
        root = 1 / math.sqrt(2)
        result = _StubResult(statevector=[complex(root), 0j, 0j, complex(root)])
        backend, client = self._backend(result)
        backend.run(self.circuit, shots=None)
        _circuit, kwargs = client.calls[0]
        self.assertNotIn("shots", kwargs)
        self.assertEqual(backend.last_warnings, [])

    def test_statevector_limit_is_configurable(self) -> None:
        result = _StubResult(counts={"00": 8})
        backend, client = self._backend(result, max_statevector_qubits=1)
        backend.run(self.circuit, shots=None)
        _circuit, kwargs = client.calls[0]
        self.assertIn("shots", kwargs)

    def test_probabilities_are_normalised_not_treated_as_counts(self) -> None:
        """Without shots the platform returns probabilities, not integers."""
        result = _StubResult(counts={"00": 0.5, "11": 0.5})
        backend, _client = self._backend(result, max_statevector_qubits=0)
        got = backend.run(self.circuit, shots=None)
        self.assertAlmostEqual(got.expectations_z[0], 0.0)
        self.assertIsNone(got.counts, "probabilities must not masquerade as counts")

    def test_truncated_distribution_is_flagged(self) -> None:
        """get_counts() returns at most 2**17 outcomes; say so rather than lie."""
        from ultraquant.quantum.bluequbit import _COUNTS_LIMIT

        counts = {format(i, "020b"): 1 for i in range(_COUNTS_LIMIT)}
        result = _StubResult(counts=counts)
        backend, _client = self._backend(result, max_statevector_qubits=0)
        backend.run(Circuit(20).h(0), shots=1024)
        self.assertTrue(any("truncated" in w for w in backend.last_warnings))

    def test_run_does_not_secretly_probe_bit_order(self) -> None:
        """verify_bit_order costs a billable job, so run() must never call it."""
        result = _StubResult(counts={"00": 512, "11": 512})
        backend, client = self._backend(result, max_statevector_qubits=0)
        backend.run(self.circuit, shots=64)
        self.assertEqual(len(client.calls), 1, "run() submitted more than one job")

    def test_verify_bit_order_detects_little_endian(self) -> None:
        result = _StubResult(counts={"01": 64})
        backend, _client = self._backend(result)
        self.assertEqual(backend.verify_bit_order(), "little")

    def test_verify_bit_order_detects_big_endian(self) -> None:
        result = _StubResult(counts={"10": 64})
        backend, _client = self._backend(result)
        self.assertEqual(backend.verify_bit_order(), "big")
        # And the ordering it found must actually be used.
        got = BlueQubitBackend._expectations_from_counts({"10": 64}, 2, "big")
        self.assertAlmostEqual(got[0], -1.0)

    def test_ambiguous_probe_is_an_error(self) -> None:
        result = _StubResult(counts={"00": 32, "11": 32})
        backend, _client = self._backend(result)
        with self.assertRaises(BackendUnavailable):
            backend.verify_bit_order()

    def test_bad_bit_order_rejected(self) -> None:
        _install_stub()
        with self.assertRaises(ValueError):
            BlueQubitBackend(api_token="t", bit_order="sideways")

    def test_failed_run_status_is_an_error(self) -> None:
        result = _StubResult(counts={"00": 1})
        result.run_status = "NOT_ENOUGH_FUNDS"
        backend, _client = self._backend(result, max_statevector_qubits=0)
        with self.assertRaises(BackendUnavailable) as caught:
            backend.run(self.circuit, shots=16)
        self.assertIn("NOT_ENOUGH_FUNDS", str(caught.exception))

    def test_client_prefers_the_documented_init(self) -> None:
        """The Quick Start uses init(); the API reference documents BQClient."""
        used = {}
        module = types.ModuleType("bluequbit")
        client = _StubClient(result=_StubResult(counts={"00": 1}))

        def fake_init(token):
            used["init"] = token
            return client

        module.init = fake_init
        module.BQClient = lambda **kw: used.setdefault("bqclient", True) or client
        sys.modules["bluequbit"] = module
        BlueQubitBackend(api_token="tok")
        self.assertEqual(used.get("init"), "tok")
        self.assertNotIn("bqclient", used)

    def test_estimate_returns_cost(self) -> None:
        backend, _client = self._backend(_StubResult(counts={"00": 1}))
        estimate = backend.estimate(self.circuit)
        self.assertEqual(estimate["estimated_cost"], 0.02)

    def test_describe_never_leaks_the_token(self) -> None:
        backend, _client = self._backend(_StubResult(counts={"00": 1}))
        import json

        blob = json.dumps(backend.describe())
        self.assertNotIn("t", json.loads(blob).get("job_name", "").replace("ultraquant", ""))
        self.assertNotIn("api_token", blob)
        self.assertTrue(backend.describe()["authenticated"])


if __name__ == "__main__":
    unittest.main()
