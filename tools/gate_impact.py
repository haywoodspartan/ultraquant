"""Which exams could a planned change flip? Read before freezing a pre-registration.

Usage:
    python tools/gate_impact.py ultraquant/distill/file.py ultraquant/distill/sources.py

For each changed module, list the gate modules (ultraquant/experiments/*_gate.py)
that import it - directly or through another experiments module they import -
and flag the ones that compare whole records (stash entries, filed claims,
replies), which pin every field a change might add. The ledger's verdict is
shown beside each, so an expected flip can be written into the criteria
before any code (see the gate-flips-before-freezing lesson: §11.149, §11.165).
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "ultraquant" / "experiments"
LEDGER = EXPERIMENTS / "records" / "gate_ledger.json"
WHOLE = re.compile(r"_comparable|_committed\b|stash\.entries\(\)\s*[!=]=|entries\(\)\)\s*[!=]=|"
                   r"== reply|reply ==|!= reply|MEASURED")


def _module(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def impact(changed: list[str]) -> list[dict]:
    targets = {_module((ROOT / c).resolve()) for c in changed}
    graph = {_module(p): _imports(p) for p in EXPERIMENTS.glob("*.py")}
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
    rows = []
    for gate in sorted(m for m in graph if m.endswith("_gate")):
        seen, stack, hits = set(), [gate], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for name in graph.get(current, ()):
                if any(name == t or name.startswith(t + ".") for t in targets):
                    hits.add(name)
                if name in graph and name not in seen:
                    stack.append(name)
        if hits:
            short = gate.rsplit(".", 1)[-1]
            text = (EXPERIMENTS / f"{short}.py").read_text(encoding="utf-8")
            entry = ledger.get("gates", {}).get(short) or {}
            rows.append({"gate": short, "via": sorted(hits)[:3],
                         "whole-record comparison": bool(WHOLE.search(text)),
                         "ledger": entry.get("verdict") or ("skipped" if short in ledger.get("skipped", {}) else "-")})
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = impact(sys.argv[1:])
    for row in rows:
        flag = "  WHOLE-RECORD" if row["whole-record comparison"] else ""
        print(f"{row['gate']:<28} {row['ledger']:<8} via {', '.join(row['via'])}{flag}")
    print(f"{len(rows)} gate(s) import the changed modules; "
          f"{sum(r['whole-record comparison'] for r in rows)} compare whole records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
