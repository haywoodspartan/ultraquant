"""Point it at a checkpoint; it says what converting would cost.

    python -m ultraquant.convert <model.gguf> [--matrices 12]

The report is the product. Any kit can turn weights into {-1, 0, +1};
what a person deciding whether to convert actually needs is the
number this prints - how far the ternary layer's output lands from
the trained layer's, on real tensors from their own file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultraquant.convert import gguf, ternary


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ultraquant.convert",
        description="Report what ternary conversion costs a checkpoint.")
    parser.add_argument("model", help="path to a .gguf checkpoint")
    parser.add_argument("--matrices", type=int, default=12,
                        help="how many weight tensors to sample")
    parser.add_argument("--rows", type=int, default=64,
                        help="rows per tensor")
    parser.add_argument("--probes", type=int, default=4,
                        help="Gaussian probes per measurement")
    args = parser.parse_args(argv)

    opened = gguf.read(Path(args.model))
    census = opened.type_census()
    readable = [info for info in opened.tensors if info.readable
                and len(info.dims) >= 2 and info.rows > 1]
    refused = {name: count for name, count in census.items()
               if name not in ("F32", "F16", "BF16", "Q8_0")}

    print(f"checkpoint : {Path(args.model).name}")
    print(f"architecture: {opened.metadata.get('general.architecture')}")
    print(f"tensors    : {len(opened.tensors)}  {census}")
    if refused:
        print(f"NOT READ   : {refused} - this reader handles F32, F16, "
              "BF16 and Q8_0, and refuses to guess at the rest")
    if not readable:
        print("nothing convertible to measure")
        return 1

    print()
    print(f"{'tensor':<38}{'error':>8}{'cosine':>8}{'rule':>18}")
    total_error = 0.0
    total_cosine = 0.0
    measured = 0
    for info in readable[:args.matrices]:
        rows = opened.rows_of(info, 0, min(args.rows, info.rows))
        if not rows or not rows[0]:
            continue
        quantised, scales, picked = ternary.choose_rules(rows)
        error, cosine = ternary.output_error(rows, quantised, scales,
                                             probes=args.probes,
                                             seed=977)
        total_error += error
        total_cosine += cosine
        measured += 1
        print(f"{info.name[:37]:<38}{error:8.4f}{cosine:8.4f}"
              f"{picked:>18}")
    if not measured:
        return 1
    print()
    print(f"mean relative output error : {total_error / measured:.4f}")
    print(f"mean cosine to the original: {total_cosine / measured:.4f}")
    print()
    print("Read that as what it is. A ternary layer whose output sits at")
    print("this cosine to the trained one is not a drop-in replacement;")
    print("the error compounds through a stack. Conversion is a starting")
    print("point for retraining, not a substitute for it.")
    return 0


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(main())
