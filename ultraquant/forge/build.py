"""CLI: forge a model from scratch.

Examples::

    python -m ultraquant.forge.build
    python -m ultraquant.forge.build --synthetic 64 --labels 4 --epochs 40
    python -m ultraquant.forge.build --tier cpu --compare

``--compare`` forges the same model on every available tier and prints the
timings side by side, which is the honest way to see what the GPU is worth for
a given library size.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultraquant.forge.corpus import (
    BUILTIN_KEYWORDS,
    build_corpora,
    builtin_taxonomy,
    synthetic_taxonomy,
)
from ultraquant.forge.forge import ModelForge
from ultraquant.forge.trainer import forge_tier_report


def main(argv: list[str] | None = None) -> int:
    """Forge a model.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Forge an UltraQuant model from scratch")
    parser.add_argument("--root", default="./forged_model", help="output directory")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="invent N categories instead of using the built-in glyphs")
    parser.add_argument("--labels", type=int, default=4,
                        help="prototypes per synthetic category")
    parser.add_argument("--per-class", type=int, default=40, help="variants per prototype")
    parser.add_argument("--flips", type=int, default=2, help="pixels flipped per variant")
    parser.add_argument("--hidden", type=int, default=32, help="hidden width per expert")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tier", default="auto",
                        choices=["auto", "both", "cuda", "cpu", "python"])
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU worker threads (0 = all cores)")
    parser.add_argument("--compare", action="store_true",
                        help="forge on every available tier and compare timings")
    parser.add_argument("--keep", action="store_true",
                        help="keep per-tier output directories when comparing")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass

    tiers = forge_tier_report()
    print("UltraQuant model forge")
    print(f"  tiers available: {tiers}")

    if args.synthetic > 0:
        taxonomy = synthetic_taxonomy(args.synthetic, args.labels, seed=args.seed)
        keywords = None
        print(f"  taxonomy: {args.synthetic} invented categories x {args.labels} prototypes")
    else:
        taxonomy = builtin_taxonomy()
        keywords = BUILTIN_KEYWORDS
        print(f"  taxonomy: {len(taxonomy)} built-in glyph families")

    corpora = build_corpora(taxonomy, n_per_class=args.per_class,
                            flips=args.flips, seed=args.seed, keywords=keywords)
    samples = sum(len(c) for c in corpora)
    print(f"  corpus:   {samples:,} augmented samples "
          f"({args.per_class} variants x {args.flips}-pixel noise)")

    def run(tier: str, root: Path):
        forge = ModelForge(root, seed=args.seed, hidden=args.hidden, tier=tier,
                           n_threads=args.threads)
        return forge, forge.build(corpora, epochs=args.epochs, lr=args.lr,
                                  batch_size=args.batch)

    if args.compare:
        print(f"\n  {'tier':<8} {'train s':>10} {'total s':>10} {'accuracy':>10}")
        results = {}
        for tier in ("python", "cpu", "cuda"):
            if tier != "python" and not tiers.get(tier):
                print(f"  {tier:<8} {'unavailable':>10}")
                continue
            if tier == "python" and samples * args.epochs > 400_000:
                print(f"  {tier:<8} {'skipped':>10}  (too slow at this size)")
                continue
            root = Path(args.root).with_name(Path(args.root).name + f"_{tier}")
            shutil.rmtree(root, ignore_errors=True)
            _forge, report = run(tier, root)
            results[tier] = report
            print(f"  {report.tier:<8} {report.train_seconds:>10.3f} "
                  f"{report.total_seconds:>10.3f} {report.mean_accuracy:>10.3f}")
            if not args.keep:
                shutil.rmtree(root, ignore_errors=True)
        if "cpu" in results and "cuda" in results:
            speedup = results["cpu"].train_seconds / results["cuda"].train_seconds
            print(f"\n  GPU vs CPU on this library: {speedup:.2f}x")
        if "python" in results and "cuda" in results:
            speedup = results["python"].train_seconds / results["cuda"].train_seconds
            print(f"  GPU vs pure Python:         {speedup:.0f}x")
        return 0

    forge, report = run(args.tier, Path(args.root))
    print(f"\n  {report.summary()}")
    dispatch = getattr(forge, "last_dispatch", None)
    if dispatch:
        line = f"  dispatch: {dispatch['config']} ({dispatch['reason']})"
        timings = dispatch.get("timings_ms")
        if timings:
            line += "  " + " ".join(f"{name}={ms}ms"
                                    for name, ms in timings.items())
        print(line)
    print(f"  store:    {report.store_bytes:,} bytes")
    if report.library:
        print(f"  library:  {report.library}")
    if report.snapshot:
        print(f"  snapshot: {report.snapshot}")

    served = forge.verify(corpora)
    mean_served = sum(served.values()) / max(len(served), 1)
    print(f"\n  Verified by paging every expert back out of the library: "
          f"mean accuracy {mean_served:.3f}")
    for name, accuracy in sorted(served.items())[:8]:
        print(f"    {name:<16} {accuracy:.3f}")
    if len(served) > 8:
        print(f"    ... and {len(served) - 8} more")

    resident = forge.cache.stats()
    print(f"\n  Paged in during verification: {len(resident['resident'])} of "
          f"{report.experts} experts, peak {resident['peak_bytes']:,} bytes "
          f"of a {resident['budget_bytes']:,} byte budget.")
    print("  The model is a library on disk; only what is used is ever resident.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
