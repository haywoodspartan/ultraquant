"""UltraQuant demo CLI.

Run from the project root::

    python -m ultraquant.demo                 # train + recognize + archive round trip
    python -m ultraquant.demo --benchmark     # hybrid-exact vs hybrid-shots vs classical

The default run picks its quantum backend with ``auto_backend()`` — a real IBM
QPU when qiskit is installed and a saved account exists, the built-in exact
statevector simulator otherwise — and prints which one it got.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from ultraquant.archive.artchive import ArTchive
from ultraquant.pattern.recognition import (
    LABELS,
    PATTERNS,
    PatternRecognizer,
    render,
)
from ultraquant.quantum.backend import Backend, SimulatorBackend, auto_backend


def _noisy_glyph(name: str, flips: int, rng: random.Random) -> list[float]:
    """Return the flat glyph for ``name`` with ``flips`` distinct pixels flipped."""
    flat = render(PATTERNS[name])
    for pix in rng.sample(range(len(flat)), flips):
        flat[pix] = 1.0 - flat[pix]
    return flat


def _probe_features(rec: PatternRecognizer) -> list[list[float]]:
    """Feature vectors for every base glyph (used to verify snapshot restores).

    Computed once and reused for both networks so that shot-sampling noise in
    the quantum layer cannot mask (or fake) a weight-restore mismatch — the
    snapshot guarantee is identical predictions given identical features.
    """
    return [rec.features(render(PATTERNS[name])) for name in LABELS]


def default_run(args: argparse.Namespace) -> int:
    """Train a recognizer, recognize noisy glyphs, and round-trip an ArTchive snapshot."""
    shots = None if args.shots == 0 else args.shots

    backend: Backend | None = None
    if args.mode == "hybrid":
        backend = auto_backend(seed=args.seed)
        print(f"[backend] auto_backend() selected: {backend.name}")
        if backend.name == "ibm-qpu":
            print("[backend] real QPU dispatch: qiskit + saved IBM account found")
        else:
            print(
                "[backend] no QPU reachable -> built-in exact statevector simulator"
            )
    else:
        print("[backend] classical mode: quantum layer disabled")

    rec = PatternRecognizer(
        mode=args.mode, backend=backend, shots=shots, seed=args.seed
    )
    print(
        f"[train] mode={args.mode} epochs={args.epochs} shots={shots} "
        f"seed={args.seed}"
    )
    stats = rec.train(epochs=args.epochs, seed=args.seed + 1)
    print(
        f"[train] final_loss={stats['final_loss']:.4f} "
        f"train_accuracy={stats['train_accuracy']:.3f}"
    )
    eval_acc = rec.evaluate()
    print(f"[eval]  accuracy={eval_acc:.3f} (fresh noisy dataset, 3 flips)")

    rng = random.Random(args.seed + 4242)
    for name in ("plus", "diamond"):
        label, conf = rec.recognize(_noisy_glyph(name, 2, rng))
        print(f"[recognize] noisy {name!r:12} -> {label!r} (confidence {conf:.3f})")

    print(f"[memory] stats: {rec.memory.stats()}")

    archive = ArTchive("./artchive_store")
    probe_feats = _probe_features(rec)
    before = [rec.net.predict(f) for f in probe_feats]
    version_id = archive.commit(f"demo-{args.mode}", rec.snapshot())
    restored_payload = archive.restore(version_id)

    fresh = PatternRecognizer(
        mode=args.mode,
        backend=SimulatorBackend(args.seed) if args.mode == "hybrid" else None,
        shots=None,
        seed=args.seed,
    )
    fresh.restore_snapshot(restored_payload)
    after = [fresh.net.predict(f) for f in probe_feats]
    if before != after:
        print("[archive] FAIL: restored predictions differ from the originals")
        return 1
    print(
        f"[archive] snapshot {version_id} committed, restored, and verified: "
        "identical predictions on all 8 base glyphs"
    )
    return 0


def benchmark(args: argparse.Namespace) -> int:
    """Train + evaluate hybrid-exact, hybrid-sampled (512 shots), and classical."""
    print(
        f"[benchmark] epochs={args.epochs} seed={args.seed} "
        "(quantum configs use the built-in simulator for a controlled comparison)"
    )
    configs: list[tuple[str, str, int | None]] = [
        ("hybrid-exact", "hybrid", None),
        ("hybrid-shots512", "hybrid", 512),
        ("classical", "classical", None),
    ]
    results: dict[str, dict[str, float]] = {}
    for name, mode, shots in configs:
        backend = SimulatorBackend(args.seed) if mode == "hybrid" else None
        rec = PatternRecognizer(
            mode=mode, backend=backend, shots=shots, seed=args.seed
        )
        t0 = time.perf_counter()
        stats = rec.train(epochs=args.epochs, seed=args.seed + 1)
        eval_acc = rec.evaluate()
        elapsed = time.perf_counter() - t0
        results[name] = {
            "train_acc": stats["train_accuracy"],
            "eval_acc": eval_acc,
            "seconds": elapsed,
        }
        print(f"[benchmark] {name}: done in {elapsed:.1f}s")

    print()
    print(f"{'config':<17} {'train_acc':>9} {'eval_acc':>9} {'seconds':>8}")
    for name, _, _ in configs:
        r = results[name]
        print(
            f"{name:<17} {r['train_acc']:>9.3f} {r['eval_acc']:>9.3f} "
            f"{r['seconds']:>8.1f}"
        )
    hybrid_acc = results["hybrid-exact"]["eval_acc"]
    classical_acc = results["classical"]["eval_acc"]
    ratio = classical_acc / hybrid_acc if hybrid_acc > 0 else float("inf")
    verdict = "PASS" if ratio >= 0.5 else "FAIL"
    print()
    print(
        f"classical_acc / hybrid_acc = {classical_acc:.3f} / {hybrid_acc:.3f} "
        f"= {ratio:.3f}  (>= 0.5 effectiveness requirement: {verdict})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0 on success)."""
    parser = argparse.ArgumentParser(
        prog="python -m ultraquant.demo",
        description="UltraQuant hybrid quantum/classical pattern-recognition demo.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "classical"),
        default="hybrid",
        help="feature pipeline for the default run (default: hybrid)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="training epochs (default: 25)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="quantum-layer shots; 0 means exact expectations (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed (default: 0)")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run the hybrid-exact / hybrid-shots / classical comparison",
    )
    args = parser.parse_args(argv)

    if args.benchmark:
        return benchmark(args)
    return default_run(args)


if __name__ == "__main__":
    sys.exit(main())
