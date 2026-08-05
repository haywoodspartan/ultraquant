# UltraQuant

UltraQuant is a small **hybrid quantum/classical pattern-recognition model** written
entirely from scratch in pure Python stdlib — no numpy, no ML frameworks. It
classifies noisy 5x5 pixel glyphs (8 classes: plus, cross, square, diamond, arrows,
stripes) with a ternary-weight neural network whose input is augmented by a quantum
feature map. The quantum layer runs on a real IBM QPU when one is reachable and on
the built-in exact statevector simulator otherwise.

To be clear about scope: **this is a compact, educational hybrid-ML demo — it is not
AGI, and it does not claim quantum advantage.** See [Effectiveness](#effectiveness-honest-numbers).
While it could claim quantim advantage. QPU access even from IBM is extremely expensive.

## Architecture

```
ultraquant/
  quantum/     from-scratch qubit engine
    gates.py      gate matrices (H, X, Y, Z, S, T, RX/RY/RZ) as list[list[complex]]
    state.py      dense statevector simulator (little-endian: qubit 0 = LSB)
    circuit.py    backend-agnostic gate-list Circuit with chainable builders
    backend.py    SimulatorBackend (exact or shot-sampled), QPUBackend (IBM Runtime
                  Estimator, guarded qiskit imports), auto_backend() dispatch
    qlayer.py     QuantumFeatureMap: RY angle encoding (pi*tanh(f)), CNOT entangler
                  ring, data re-uploading; outputs per-qubit <Z> in [-1, 1]
  model/       ultra-quantized neural network
    quantize.py   ternary weight quantization ({-1,0,+1} + one fp scale, straight-
                  through estimator), activation fake-quantization, TernaryLinear
    network.py    UltraQuantNet: MLP with ternary hidden layers, fp head, softmax
                  cross-entropy SGD, JSON-safe state_dict round trips
  memory/      systematic memory
    systematic.py episodic log + bounded FIFO working memory + semantic facts with
                  reinforcement/revision + bit-signature index (Hamming similarity),
                  all JSON-persistable
  archive/     the Ar(T)chive
    artchive.py   **T**emporal archive of model ar**T**ifacts — hence Ar(T)chive:
                  T-numbered snapshots (T-0001, T-0002, ...), sha256 tamper-evident
                  manifest, restore-time integrity verification, top-level diffs
  pattern/     the glue
    recognition.py built-in glyph dataset, PatternRecognizer (hybrid or classical
                  feature pipeline -> UltraQuantNet -> memory logging), snapshots
  demo.py      CLI: train/recognize/archive round trip, and a benchmark
```

**How a recognition flows:** a glyph's 25 pixels are kept as-is, its 5 per-row means
are squashed to rotation angles and encoded into a 5-qubit circuit (RY layer + CNOT
ring, re-uploaded in chunks), and the measured/computed Z expectations become 5 extra
features. The 30-dim vector goes through the ternary MLP; the prediction is logged as
an episode in systematic memory and the glyph's bit signature is indexed under the
predicted label. Model snapshots are committed to the Ar(T)chive, whose manifest
sha256s make any on-disk tampering detectable at restore time.

## Quickstart

From the project root (Python 3.13, no dependencies to install):

```
# full test suite (~2 s)
python -m unittest discover -s tests -v

# train a hybrid recognizer, recognize noisy glyphs, archive round trip
python -m ultraquant.demo --epochs 25

# classical-only pipeline
python -m ultraquant.demo --mode classical

# shot-sampled quantum layer (finite-shot noise, like real hardware)
python -m ultraquant.demo --shots 512

# hybrid-exact vs hybrid-512-shots vs classical comparison
python -m ultraquant.demo --benchmark
```

## QPU vs. simulator dispatch

`auto_backend()` in `ultraquant/quantum/backend.py` is the single dispatch point:

- If `qiskit` + `qiskit_ibm_runtime` are installed **and** a saved IBM Quantum
  account exists, it returns `QPUBackend`, which translates UltraQuant circuits to
  Qiskit and estimates Z expectations on real hardware via the Runtime Estimator
  primitive (default 1024 shots).
- Otherwise it silently falls back to `SimulatorBackend`, the built-in pure-Python
  dense statevector engine. It never raises.

All qiskit imports are guarded inside `QPUBackend` methods, so the package — and its
whole test suite — works identically with or without qiskit installed. Tests always
use the simulator explicitly and never touch a QPU.

## Effectiveness (honest numbers)

Measured with `python -m ultraquant.demo --benchmark` (25 epochs, seed 0, 320
training / 120 evaluation samples, evaluation glyphs corrupted with 3 pixel flips):

| config           | train acc | eval acc |
|------------------|-----------|----------|
| hybrid-exact     | 0.969     | 0.933    |
| hybrid-shots512  | 0.963     | 0.933    |
| classical        | 0.972     | 0.917    |

**Ratio: classical / hybrid-exact = 0.917 / 0.933 = 0.982**, far above the 0.5
("classical must retain at least 50% of hybrid effectiveness") requirement.

What these numbers honestly mean:

- The hybrid and classical pipelines perform **essentially the same** on this task.
  The quantum feature map is a nonlinear transform of 5 row means; the classical
  pipeline feeds the same row means directly, and the ternary MLP learns the task
  either way. There is no quantum advantage here, and none is claimed — the task is
  small, the encoding is simple, and 25 raw pixels already carry most of the signal.
- A traditional machine loses nothing by not having a QPU: the classical statevector
  simulator computes **exact** expectation values, whereas a real QPU can only
  *estimate* them from finite shot samples (plus hardware noise). The
  `hybrid-shots512` row shows the shot-sampled regime — on this task even 512-shot
  estimation matches the exact result, and exact simulation is strictly the cleaner
  signal.
- What the project *does* demonstrate: a working end-to-end hybrid architecture — a
  correct little-endian statevector engine, angle encoding with data re-uploading,
  ternary weight quantization trained with a straight-through estimator, systematic
  memory, and tamper-evident model archiving — in ~2,500 lines of dependency-free
  Python, with the same code path able to target real IBM hardware.

UltraQuant is a hybrid quantum/classical ML demo. It is not AGI, and its name
describes its quantization scheme, not its intelligence for now...
