# UltraQuant — Specification (v0.1)

UltraQuant is a hybrid quantum/classical pattern-recognition model, implemented **entirely
from scratch in pure Python stdlib** (no numpy, no third-party packages). The single
allowed third-party import is `qiskit` / `qiskit_ibm_runtime`, and only inside a guarded
`try/except` in `QPUBackend` — the package must work perfectly when they are absent.

Target: Python 3.13 on Windows. All paths handled with `pathlib` or `os.path`.
All randomness goes through `random.Random(seed)` instances — never the module-level
`random` functions — so every component is deterministic when seeded.

Layout (rooted at `H:\AI Model AGI`):

```
ultraquant/
  __init__.py                 (exists — do not modify)
  quantum/
    __init__.py               (exists — do not modify)
    gates.py
    state.py
    circuit.py
    backend.py
    qlayer.py
  model/
    __init__.py               (exists — do not modify)
    quantize.py
    network.py
  memory/
    __init__.py               (exists — do not modify)
    systematic.py
  archive/
    __init__.py               (exists — do not modify)
    artchive.py
  pattern/
    __init__.py               (exists — do not modify)
    recognition.py
  demo.py
tests/
  __init__.py                 (exists — do not modify)
  test_quantum.py
  test_qlayer.py
  test_model.py
  test_memory.py
  test_archive.py
  test_pattern.py
SPEC.md                       (this file)
README.md
```

Run tests from the root: `python -m unittest tests.test_quantum -v` etc.
Every public class/function gets a docstring and type hints.

---

## 1. `ultraquant/quantum/gates.py`

Gate matrices as `list[list[complex]]` (2x2 unless noted).

- Constants: `H`, `X`, `Y`, `Z`, `S`, `T`, `I2` — standard matrices
  (`H = 1/sqrt(2) * [[1,1],[1,-1]]`, etc.).
- `def rx(theta: float) -> Matrix`: `[[cos(t/2), -1j*sin(t/2)], [-1j*sin(t/2), cos(t/2)]]`
- `def ry(theta: float) -> Matrix`: `[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]`
- `def rz(theta: float) -> Matrix`: `[[exp(-1j*t/2), 0], [0, exp(1j*t/2)]]`
- `Matrix = list[list[complex]]` type alias exported.

## 2. `ultraquant/quantum/state.py`

`class QuantumState` — dense statevector simulator.

- `__init__(self, num_qubits: int, seed: int | None = None)` — state `|0...0>`:
  `self.amplitudes: list[complex]` of length `2**num_qubits`, index bit convention
  **little-endian: qubit q is bit q (LSB = qubit 0) of the basis-state index**.
  Internal RNG: `random.Random(seed)`.
- `apply_gate(self, matrix: Matrix, qubit: int) -> None` — apply 2x2 gate to `qubit`
  (iterate index pairs differing only in bit `qubit`).
- `apply_controlled(self, matrix: Matrix, control: int, target: int) -> None` —
  apply gate to `target` only where the `control` bit is 1. (CNOT = controlled X,
  CZ = controlled Z.)
- `probabilities(self) -> list[float]` — `|amp|**2` per basis index.
- `expectation_z(self, qubit: int) -> float` — exact `<Z_q>` = sum of p(i) * (+1 if
  bit q of i == 0 else -1).
- `measure_all(self, shots: int) -> dict[str, int]` — sample basis states from
  `probabilities()` with the internal RNG. Keys are bitstrings with **qubit 0 as the
  rightmost character** (e.g. for 2 qubits, index 1 → `"01"`).
- `norm(self) -> float` — L2 norm of amplitudes (for tests).

## 3. `ultraquant/quantum/circuit.py`

- `@dataclass(frozen=True) class Op: name: str; qubits: tuple[int, ...]; params: tuple[float, ...] = ()`
- `class Circuit`:
  - `__init__(self, num_qubits: int)`; `self.ops: list[Op]`.
  - Builder methods (each appends an Op and returns `self` for chaining):
    `h(q)`, `x(q)`, `y(q)`, `z(q)`, `s(q)`, `t(q)`, `rx(q, theta)`, `ry(q, theta)`,
    `rz(q, theta)`, `cnot(control, target)`, `cz(control, target)`,
    `swap(a, b)` (emit as three CNOTs or a dedicated op — your choice, but the
    simulator must honor whatever you emit).
  - `apply_to(self, state: QuantumState) -> None` — replay ops onto a state.

## 4. `ultraquant/quantum/backend.py`

- `class BackendUnavailable(RuntimeError)`
- `@dataclass class BackendResult: expectations_z: list[float]; counts: dict[str, int] | None`
- `class Backend(abc.ABC)`: attribute `name: str`;
  `run(self, circuit: Circuit, shots: int | None = None) -> BackendResult`.
- `class SimulatorBackend(Backend)` — `name = "ultraquant-statevector"`.
  `__init__(self, seed: int | None = None)`. `run`: build `QuantumState` (seeded from
  an internal per-run counter + base seed so repeated runs are deterministic but not
  identical draws), apply circuit. If `shots is None`: exact `expectations_z` for every
  qubit, `counts=None`. If `shots` given: sample `measure_all(shots)` and *estimate*
  `expectations_z[q]` from the sampled counts; also return the counts.
- `class QPUBackend(Backend)` — `name = "ibm-qpu"`. Constructor takes optional
  `backend_name: str | None`. `classmethod available() -> bool`: True only if
  `qiskit` + `qiskit_ibm_runtime` import cleanly AND a saved IBM account exists;
  every failure path returns False (never raises). `run()` translates the Circuit ops
  to a Qiskit circuit and estimates Z expectations via the Runtime Estimator primitive
  (shots default 1024 when None); raise `BackendUnavailable` if anything fails.
  All qiskit imports live inside methods, guarded.
- `def auto_backend(prefer_qpu: bool = True, seed: int | None = None) -> Backend` —
  returns `QPUBackend()` if `prefer_qpu` and `QPUBackend.available()`, else
  `SimulatorBackend(seed)`. Must never raise.

## 5. `ultraquant/quantum/qlayer.py`

`class QuantumFeatureMap` — the quantum layer of the model.

- `__init__(self, num_qubits: int, backend: Backend, shots: int | None = None)`.
- `encode(self, features: list[float]) -> Circuit`:
  - Angles: `theta_i = pi * tanh(f_i)` for numerical safety.
  - Layer 1: `ry(q, theta_q)` for each qubit q using the first `num_qubits` features
    (missing features → angle 0).
  - Entangler: ring of CNOTs `cnot(q, (q+1) % num_qubits)` (skip ring for 1 qubit).
  - Data re-uploading: if more than `num_qubits` features remain, repeat
    (RY layer with the next chunk, then entangler ring) until features are exhausted.
- `extract(self, features: list[float]) -> list[float]` — `backend.run(encode(...), shots)`,
  return `expectations_z` (length `num_qubits`, each in [-1, 1]).

## 6. `ultraquant/model/quantize.py`

Ternary Weight Network-style quantization (values in {-1, 0, +1} + one fp scale).

- `def quantize_ternary(weights: list[list[float]]) -> tuple[list[list[int]], float]`:
  `delta = 0.75 * mean(|w|)` over all entries (delta=0 edge case → all zeros, scale 1.0);
  `q_ij = sign(w_ij) if |w_ij| > delta else 0`;
  `alpha = mean(|w_ij|)` over entries where `q_ij != 0` (or 1.0 if none).
  Returns `(q, alpha)`.
- `def dequantize(q: list[list[int]], alpha: float) -> list[list[float]]`.
- `def fake_quantize_activation(x: list[float], bits: int = 8, clip: float = 4.0) -> list[float]`:
  clamp to [-clip, clip], round to `2**bits - 1` uniform levels across that range,
  return the dequantized floats.
- `class TernaryLinear`:
  - `__init__(self, in_dim: int, out_dim: int, rng: random.Random, quantized: bool = True)`;
    master weights `self.w: list[list[float]]` shape `[out_dim][in_dim]` init
    uniform(-r, r) with `r = sqrt(6 / (in_dim + out_dim))`; bias `self.b: list[float]`
    zeros (bias stays full-precision).
  - `forward(self, x: list[float]) -> list[float]` — if `quantized`, use
    `alpha * q` weights (recompute from master each call — fine at this scale);
    else use master weights directly. Caches last input + last quantized weights
    for backward.
  - `backward(self, grad_out: list[float]) -> list[float]` — returns grad wrt input
    computed with the *same* (quantized) weights used in forward; accumulates
    `self.gw`, `self.gb` gradients **for the master weights** via the
    straight-through estimator (i.e., gradient of the quantized weight is passed
    straight to the master weight).
  - `step(self, lr: float)` — SGD update on master weights and bias, then clamp
    master weights to [-1, 1]; zero accumulated grads.
  - `state_dict() -> dict` / `load_state_dict(d)` — JSON-safe (lists + floats).

## 7. `ultraquant/model/network.py`

`class UltraQuantNet` — MLP, ternary hidden layers, full-precision head by default.

- `__init__(self, input_dim: int, hidden_dims: list[int], num_classes: int,
  seed: int = 0, quantize_head: bool = False, act_bits: int = 8)`.
  Hidden layers: `TernaryLinear` + ReLU + `fake_quantize_activation`.
  Head: `TernaryLinear(..., quantized=quantize_head)`.
- `forward(self, x: list[float]) -> list[float]` — logits.
- `predict(self, x) -> tuple[int, float]` — argmax label + softmax confidence.
- `train_batch(self, xs: list[list[float]], ys: list[int], lr: float) -> float` —
  forward+backward each sample (softmax cross-entropy; grad = softmax - onehot),
  accumulate, one `step(lr / len(xs))`-equivalent update (your choice: scale grads
  or lr, just be consistent), return mean loss.
- Numerical safety: softmax with max-subtraction; cross-entropy with epsilon clamp.
- `state_dict() / load_state_dict()` — JSON-safe round trip that reproduces
  identical predictions.

## 8. `ultraquant/memory/systematic.py`

`class SystematicMemory` — three stores + bit-signature index, JSON persistence.

- `__init__(self, path: str | os.PathLike | None = None, working_capacity: int = 16)`;
  if `path` given and exists → auto-`load()`.
- Episodic: `remember_episode(kind: str, content: dict, tags: list[str] | None = None) -> int`
  — appends `{id, t, kind, content, tags}` (t = `datetime.now(timezone.utc).isoformat()`),
  returns id (incrementing int). Also pushes the id into working memory (bounded
  FIFO of `working_capacity`).
- Semantic: `remember_fact(key: str, value, confidence: float = 0.5) -> None` —
  new key → store `{value, confidence, reinforcements: 0, first_seen, last_seen}`;
  same key + equal value → `confidence = min(1.0, confidence + 0.1)`,
  `reinforcements += 1`; same key + different value → replace value, reset
  confidence to the given value, and log a `"revision"` episode.
  `recall_fact(key) -> dict | None`.
- Signatures: `store_signature(label: str, bits: list[int]) -> None`;
  `nearest_signature(bits: list[int]) -> tuple[str, float] | None` — max similarity
  `1 - hamming/len` over stored signatures (compare only equal-length signatures).
- `recall_episodes(kind: str | None = None, tags: list[str] | None = None, limit: int = 10)`
  — most recent first; tag filter = any-overlap.
- `working() -> list[dict]` — episodes currently in working memory, oldest first.
- `save() / load()`; `stats() -> dict` (counts per store).

## 9. `ultraquant/archive/artchive.py`

`class ArTchive` — the Ar(T)chive: T-numbered, tamper-evident temporal snapshots.

- `class IntegrityError(RuntimeError)`
- `__init__(self, root: str | os.PathLike)` — creates `root/` and `root/snapshots/`;
  manifest at `root/manifest.json` (`{"versions": [...]}`), created if missing.
- `commit(self, label: str, payload: dict) -> str` — next id `"T-0001"`, `"T-0002"`, ...;
  serialize payload with `json.dumps(payload, sort_keys=True, separators=(",", ":"))`,
  write to `snapshots/{id}.json` (utf-8), record
  `{id, label, timestamp, sha256, path}` in the manifest (sha256 of the exact bytes
  written). Returns the id.
- `restore(self, version_id: str) -> dict` — read the snapshot bytes, recompute
  sha256, mismatch with manifest → raise `IntegrityError`; else parsed payload.
- `versions(self) -> list[dict]` — manifest entries, oldest first. `latest() -> str | None`.
- `diff(self, a: str, b: str) -> dict` — `{"added": [...], "removed": [...], "changed": [...]}`
  comparing top-level keys of the two payloads.

## 10. `ultraquant/pattern/recognition.py`

Built-in dataset — 8 glyph classes, 5x5, `#` = 1, `.` = 0. Use EXACTLY these:

```
PATTERNS: dict[str, list[str]] = {
  "plus":      ["..#..", "..#..", "#####", "..#..", "..#.."],
  "cross":     ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
  "square":    ["#####", "#...#", "#...#", "#...#", "#####"],
  "diamond":   ["..#..", ".#.#.", "#...#", ".#.#.", "..#.."],
  "arrow_up":  ["..#..", ".###.", "#####", "..#..", "..#.."],
  "arrow_down":["..#..", "..#..", "#####", ".###.", "..#.."],
  "stripes_h": ["#####", ".....", "#####", ".....", "#####"],
  "stripes_v": ["#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
}
```
Class order/index = insertion order above. `LABELS: list[str]` derived from it.

- `def render(rows: list[str]) -> list[float]` — flatten to 25 floats (row-major).
- `def make_dataset(n_per_class: int, flips: int, seed: int) -> tuple[list[list[float]], list[int]]`
  — for each class, `n_per_class` samples: copy base glyph, flip `flips` distinct
  random pixels (`random.Random(seed)` shared across the whole dataset build).
- `def row_means(x: list[float]) -> list[float]` — mean of each of the 5 rows.
- `class PatternRecognizer`:
  - `__init__(self, mode: str = "hybrid", backend: Backend | None = None,
    shots: int | None = None, seed: int = 0, hidden: tuple[int, ...] = (32,),
    memory_path=None)`.
    `mode` in {"hybrid", "classical"}. hybrid: `backend or auto_backend(seed=seed)`,
    `QuantumFeatureMap(5, backend, shots)`. Net input dim = 30 in both modes.
  - `features(self, x: list[float]) -> list[float]` — `x` (25) + 5 extra:
    hybrid → `qmap.extract(row_means(x))`; classical → `row_means(x)` as-is.
  - `train(self, epochs: int = 40, lr: float = 0.05, n_per_class: int = 40,
    flips: int = 2, seed: int = 1) -> dict` — build dataset, precompute features
    once, shuffled minibatches of 16, return
    `{"final_loss": float, "train_accuracy": float, "epochs": int}`.
  - `evaluate(self, n_per_class: int = 15, flips: int = 3, seed: int = 99) -> float`
    — accuracy on a fresh dataset.
  - `recognize(self, rows_or_flat) -> tuple[str, float]` — accepts 5 row-strings or
    25 floats; returns `(label_name, confidence)`; logs a `"recognition"` episode
    (content: label, confidence, mode) and stores the input's bit signature under
    the predicted label in memory.
  - `snapshot(self) -> dict` — `{"config": {...}, "net": state_dict, "memory_stats": ...}`
    JSON-safe; `restore_snapshot(self, payload: dict)` — rebuild net weights
    (must reproduce identical predictions).

## 11. `ultraquant/demo.py`

CLI (argparse), runnable as `python -m ultraquant.demo` from the root:

- Default run: pick backend via `auto_backend()` (print which one — this is the
  "uses qubits on a QPU, simulator otherwise" dispatch), train a hybrid recognizer
  (modest epochs, seeded), print train/eval accuracy, `recognize()` a couple of noisy
  glyphs, show `memory.stats()`, commit a snapshot to an `ArTchive` at
  `./artchive_store`, restore it, verify identical predictions, print the T-version.
- `--mode {hybrid,classical}`, `--epochs N`, `--shots N` (0 → exact/None), `--seed N`.
- `--benchmark`: train+eval three configs — hybrid exact (shots=None),
  hybrid sampled (shots=512), classical — and print a small table plus the ratio
  `classical_acc / hybrid_acc` (the ≥50%-effectiveness check). Keep epochs moderate
  so the whole benchmark finishes in a few minutes of pure Python.
- Exit code 0 on success.

## 12. Tests (unittest, one file per area)

- `tests/test_quantum.py`: H on |0> → probs [0.5, 0.5]; Bell (H then CNOT) → only
  "00"/"11" counts, exact `<Z>` == 0.0 (±1e-9) both qubits; ry/rx/rz unitarity
  (norm preserved at random angles); little-endian convention check (X on qubit 0 of
  2 qubits → index 1 has amplitude 1, counts key "01"); seeded `measure_all`
  deterministic; SimulatorBackend shots=4096 estimate within 0.1 of exact.
- `tests/test_qlayer.py`: extract length == num_qubits, values in [-1, 1];
  exact-mode determinism; data re-uploading path (7 features on 3 qubits) works.
- `tests/test_model.py`: `quantize_ternary` on a hand-built matrix (assert exact
  q + alpha); XOR (inputs (0,0),(0,1),(1,0),(1,1) → labels 0,1,1,0, hidden (8,),
  full-precision head) trains to 4/4 accuracy within a few hundred epochs;
  state_dict round trip → identical logits.
- `tests/test_memory.py`: reinforcement raises confidence and count; conflicting
  fact logs revision episode; save/load round trip preserves stats; nearest_signature.
- `tests/test_archive.py`: commit → restore round trip equality; manual file tamper
  → `IntegrityError`; two commits → diff reports changed keys; ids increment.
- `tests/test_pattern.py`: classical mode trains (small: n_per_class=15, epochs≤30)
  to eval accuracy ≥ 0.8; hybrid exact mode likewise ≥ 0.8 (keep it small so it's
  fast); recognize logs episode + returns a known label; snapshot/restore identical
  predictions. Use SimulatorBackend explicitly (never QPU) in tests.

Tests must be self-contained, fast (whole suite well under ~3 minutes), and quiet
about qiskit (its absence must never fail anything).
