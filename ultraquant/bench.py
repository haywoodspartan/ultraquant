"""Benchmark the execution tiers: pure Python, C++ CPU, CUDA, and CPU+GPU together.

Every tier computes the same numbers; only the speed differs.  That is the point
of the design — the native tiers are an accelerator, never a change in behaviour,
so a machine with no compiler and no GPU still gets identical results, just more
slowly.

Run ``python -m ultraquant.bench`` for the default sweep.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable

from ultraquant.native import accel
from ultraquant.native.hetero import HeterogeneousFeatureExtractor
from ultraquant.quantum.backend import SimulatorBackend
from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.qlayer import QuantumFeatureMap


def _time(fn: Callable[[], object], repeat: int) -> float:
    """Best-of-``repeat`` wall-clock seconds for ``fn``."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _row(name: str, seconds: float | None, baseline: float | None, unit: str) -> str:
    """Format one result row."""
    if seconds is None:
        return f"  {name:<28} {'unavailable':>14}"
    rate = f"{seconds * 1000:.2f} ms"
    speedup = f"{baseline / seconds:6.1f}x" if baseline and seconds > 0 else "     --"
    return f"  {name:<28} {rate:>14}  {speedup:>8}  {unit}"


def bench_feature_map(samples: int, qubits: int, repeat: int) -> None:
    """Batched quantum feature-map throughput across all tiers."""
    rng = random.Random(0)
    batch = [[rng.uniform(-2, 2) for _ in range(qubits)] for _ in range(samples)]
    print(f"\n[1] Quantum feature map - {samples} samples x {qubits} qubits")

    qmap = QuantumFeatureMap(qubits, SimulatorBackend(seed=0))
    py = _time(lambda: [qmap.extract(row) for row in batch], 1)
    print(_row("pure Python", py, py, f"{samples / py:,.0f} samples/s"))

    if accel.load_cpu() is not None:
        one = _time(lambda: accel.feature_map_batch_cpu(batch, qubits, 1), repeat)
        print(_row("C++ (1 thread)", one, py, f"{samples / one:,.0f} samples/s"))
        allt = _time(lambda: accel.feature_map_batch_cpu(batch, qubits, 0), repeat)
        print(_row("C++ (all threads)", allt, py, f"{samples / allt:,.0f} samples/s"))
    else:
        print(_row("C++ (1 thread)", None, None, ""))

    if accel.gpu_available():
        gpu = _time(lambda: accel.feature_map_batch_gpu(batch, qubits), repeat)
        print(_row(f"CUDA ({accel.gpu_device_name()})", gpu, py, f"{samples / gpu:,.0f} samples/s"))
    else:
        print(_row("CUDA", None, None, ""))

    extractor = HeterogeneousFeatureExtractor(qubits)
    extractor.extract_batch(batch)  # warm up the split
    het = _time(lambda: extractor.extract_batch(batch), repeat)
    split = extractor.last_split
    print(_row("CPU+GPU together", het, py, f"{samples / het:,.0f} samples/s"))
    print(f"  {'':28} split: gpu={split['gpu']} cpu={split['cpu']} python={split['python']}")


def bench_large_circuit(qubits: int, repeat: int) -> None:
    """Single large-statevector latency: the regime where the GPU wins."""
    rng = random.Random(1)
    circuit = Circuit(qubits)
    for _ in range(8):
        for q in range(qubits):
            circuit.ry(q, rng.uniform(0, 3.14))
        for q in range(qubits):
            circuit.cnot(q, (q + 1) % qubits)
    print(f"\n[2] Single circuit - {qubits} qubits, {len(circuit.ops)} ops "
          f"({2 ** qubits:,} amplitudes)")

    backend = SimulatorBackend(seed=0)
    py = _time(lambda: backend.run(circuit), 1)
    print(_row("pure Python", py, py, ""))

    if accel.load_cpu() is not None:
        cpu = _time(lambda: accel.run_circuit_cpu(circuit), repeat)
        print(_row("C++", cpu, py, ""))
    else:
        print(_row("C++", None, None, ""))

    if accel.gpu_available():
        gpu = _time(lambda: accel.run_circuit_gpu(circuit), repeat)
        print(_row("CUDA", gpu, py, ""))
    else:
        print(_row("CUDA", None, None, ""))


def bench_ternary(samples: int, repeat: int) -> None:
    """Ternary-weight inference latency at the demo network's shape."""
    rng = random.Random(2)
    in_dim, out_dim = 30, 32
    qw = [[rng.choice((-1, 0, 1)) for _ in range(in_dim)] for _ in range(out_dim)]
    bias = [0.0] * out_dim
    xs = [[rng.uniform(-1, 1) for _ in range(in_dim)] for _ in range(samples)]
    alpha = 0.37
    print(f"\n[3] Ternary inference - {samples} samples, {in_dim}->{out_dim}")

    def python_forward() -> list[list[float]]:
        return [
            [alpha * sum(qw[o][i] * x[i] for i in range(in_dim)) + bias[o] for o in range(out_dim)]
            for x in xs
        ]

    py = _time(python_forward, 1)
    print(_row("pure Python", py, py, f"{samples / py:,.0f} samples/s"))

    if accel.load_cpu() is not None:
        cpu = _time(lambda: accel.ternary_forward_batch_cpu(qw, alpha, bias, xs, 0), repeat)
        print(_row("C++ (all threads)", cpu, py, f"{samples / cpu:,.0f} samples/s"))
    else:
        print(_row("C++ (all threads)", None, None, ""))

    if accel.gpu_available():
        gpu = _time(lambda: accel.ternary_forward_batch_gpu(qw, alpha, bias, xs), repeat)
        print(_row("CUDA", gpu, py, f"{samples / gpu:,.0f} samples/s"))
    else:
        print(_row("CUDA", None, None, ""))


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark sweep.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="UltraQuant tier benchmarks")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--big-qubits", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)

    from ultraquant.native.dispatch import tier_report

    print("UltraQuant tier benchmark")
    print(f"  tiers available: {tier_report()}")

    bench_feature_map(args.samples, args.qubits, args.repeat)
    bench_large_circuit(args.big_qubits, args.repeat)
    bench_ternary(args.samples, args.repeat)

    print("\nAll tiers compute identical values (verified to 1e-9 in "
          "tests/test_native_*.py); only the speed differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
