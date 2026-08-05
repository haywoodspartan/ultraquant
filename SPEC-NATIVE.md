# UltraQuant — Native Acceleration Tier Specification (v0.1)

Extends SPEC.md (read that first — its conventions are binding). This tier adds
optional C++ (CPU) and CUDA (GPU) acceleration behind the exact same interfaces.
The pure-Python tier remains the correctness oracle; every native path must agree
with it to 1e-9 on double precision.

Machine facts (verified): Windows 11, Python 3.13.2, RTX 4090 (24 GB),
CUDA toolkit 12.6 (`nvcc` on PATH), VS 2022 Enterprise at
`C:\Program Files\Microsoft Visual Studio\2022\Enterprise` with MSVC toolsets
14.38.33130 and 14.44.35207. `cl.exe` is NOT on PATH — build via `native\build.ps1`
(already written — do not rewrite it; call it).

Precision: double (fp64) everywhere. Numbers cross the FFI as C `double`;
statevectors as interleaved re/im (`double[2 * 2**n]`); quantized weights as
`int8_t`. All exported symbols use `extern "C"` + `__declspec(dllexport)`.

New layout (additions only):

```
native/
  build.ps1            (exists — call it, never rewrite; -Target cpu|cuda|all)
  uq_core.cpp          C++17 CPU library
  uq_cuda.cu           CUDA library
ultraquant/native/
  __init__.py          docstring only
  _bin/                build output dir (created by build.ps1): ultraquant_native.dll, ultraquant_cuda.dll
  accel.py             ctypes loaders + raw wrappers
  backends.py          NativeSimulatorBackend, CudaSimulatorBackend
  hetero.py            HeterogeneousFeatureExtractor (CPU+GPU split)
  dispatch.py          best_backend() tier selection
ultraquant/bench.py    latency/throughput benchmark CLI
tests/
  test_native_cpu.py
  test_native_cuda.py
  test_native_dispatch.py
```

Circuit wire format (shared by both DLLs): a circuit of N ops is three parallel
arrays — `int32 opcodes[N]`, `int32 qubits[2*N]` (q0, q1; unused slot = -1),
`double params[N]` (unused slot = 0.0). Opcodes:
`0=h 1=x 2=y 3=z 4=s 5=t 6=rx 7=ry 8=rz 9=cnot 10=cz 11=swap`.
Python side builds these from `Circuit.ops` in `accel.py`
(`def encode_circuit(circuit) -> tuple[array('i'), array('i'), array('d')]`).

The feature-map program (angle map `theta = pi * tanh(f)`, RY layer over a chunk of
`num_qubits` features, ring CNOT entangler, data re-uploading until features are
exhausted, then per-qubit `<Z>`) is reimplemented natively from the same definition
in SPEC.md §5 — it must match `QuantumFeatureMap.extract` exactly.

---

## 1. `native/uq_core.cpp` → `ultraquant_native.dll` (C++17, no deps)

Exports (all `extern "C"`):

- `int uq_version(void)` → `1`.
- `void uq_init_state(double* amp, int num_qubits)` — set |0...0>.
- `void uq_run_circuit(double* amp, int num_qubits, const int* opcodes, const int* qubits, const double* params, int n_ops)`
  — replay a whole circuit (little-endian convention identical to SPEC.md §2).
  Gate math hand-rolled; no `std::complex` requirement (pairs of doubles are fine).
- `void uq_probabilities(const double* amp, int num_qubits, double* out /*2**n*/)`
- `void uq_expectations_z(const double* amp, int num_qubits, double* out /*n*/)`
- `void uq_feature_map(const double* features, int n_features, int num_qubits, double* out /*num_qubits*/)`
  — single-sample feature map, start to finish, in one call.
- `void uq_feature_map_batch(const double* features /*n_samples*n_features*/, int n_samples, int n_features, int num_qubits, double* out /*n_samples*num_qubits*/, int n_threads)`
  — samples partitioned across `std::thread`s (`n_threads <= 0` → hardware_concurrency).
- `void uq_ternary_forward(const signed char* qw /*out_dim*in_dim*/, double alpha, const double* bias, const double* x, int in_dim, int out_dim, double* out)`
- `void uq_ternary_forward_batch(const signed char* qw, double alpha, const double* bias, const double* xs, int n_samples, int in_dim, int out_dim, double* out, int n_threads)`

Build check: `powershell -ExecutionPolicy Bypass -File native\build.ps1 -Target cpu`.

## 2. `native/uq_cuda.cu` → `ultraquant_cuda.dll` (CUDA 12.6)

Exports (same C ABI):

- `int uqg_available(void)` — CUDA device count; 0 on any error (must never crash).
- `int uqg_device_name(char* buf, int buflen)` — 0 on success.
- `void uqg_feature_map_batch(const double* features, int n_samples, int n_features, int num_qubits, double* out)`
  — one thread block per sample; the per-sample statevector (`2**num_qubits`
  complex doubles, num_qubits <= 10) lives in shared memory; threads in the block
  cooperate on gate application; H2D/D2H transfers inside the call.
- `void uqg_run_circuit(double* amp_host, int num_qubits, const int* opcodes, const int* qubits, const double* params, int n_ops)`
  — single large statevector in global memory (grid-stride over amplitude pairs,
  one kernel launch per gate is fine); supports num_qubits up to ~28.
- `void uqg_expectations_z_from(const double* amp_host, int num_qubits, double* out)` —
  may be computed host-side after copy-back; exactness matters, speed does not.
- `void uqg_ternary_forward_batch(const signed char* qw, double alpha, const double* bias, const double* xs, int n_samples, int in_dim, int out_dim, double* out)`

Build check: `powershell -ExecutionPolicy Bypass -File native\build.ps1 -Target cuda`.
(build.ps1 already pins `-ccbin` to MSVC 14.38 for nvcc.)

## 3. `ultraquant/native/accel.py`

- `def load_cpu() -> ctypes.CDLL | None` / `def load_gpu() -> ctypes.CDLL | None` —
  look in `ultraquant/native/_bin/`; set `argtypes/restype` on every symbol;
  cache the handle module-level; return None on any failure (missing file, load
  error, `uqg_available() == 0`). Never raise on import.
- `def encode_circuit(circuit: Circuit)` — wire format above; raise `ValueError`
  on unknown op names.
- Thin typed wrappers: `run_circuit_cpu(...)`, `feature_map_batch_cpu(...)`,
  `feature_map_batch_gpu(...)`, `ternary_forward_cpu(...)`, etc., using
  `array.array` / `ctypes` buffers (no numpy).

## 4. `ultraquant/native/backends.py`

- `class NativeSimulatorBackend(Backend)` — `name = "ultraquant-native-cpu"`.
  Same semantics as `SimulatorBackend`: exact `expectations_z` when `shots is None`;
  with shots, sample in Python (seeded `random.Random`) from `uq_probabilities`.
  Constructor `__init__(self, seed=None)`; raises `BackendUnavailable` if the CPU
  DLL is absent.
- `class CudaSimulatorBackend(Backend)` — `name = "ultraquant-cuda"`, same contract,
  built on `uqg_run_circuit`; `BackendUnavailable` if GPU tier absent.

## 5. `ultraquant/native/hetero.py`

`class HeterogeneousFeatureExtractor` — "CPU and GPU at the same time".

- `__init__(self, num_qubits: int, cpu_threads: int = 0, force: str | None = None)`
  — `force` in {None, "cpu", "gpu", "python"} for testing. Detect available tiers.
- `extract_batch(self, features_batch: list[list[float]]) -> list[list[float]]`:
  - Both tiers available: on first batch, time a small warmup chunk on each
    (GPU chunk on the calling thread is fine — CPU work runs in worker threads),
    derive throughput ratio, cache it; split every batch GPU/CPU by that ratio,
    run the GPU slice and the CPU slice **concurrently** (GPU call on a
    `threading.Thread`, CPU via `uq_feature_map_batch` with its own threads),
    join, reassemble in original order.
  - One tier: use it. No tiers: fall back to pure-Python `QuantumFeatureMap` on
    `SimulatorBackend`.
  - `last_split: dict` attribute for introspection:
    `{"gpu": n, "cpu": n, "python": n}` for the most recent batch.
- Results must be bit-identical in ordering and 1e-9-close in value to pure Python
  regardless of the split.

## 6. `ultraquant/native/dispatch.py`

- `def best_backend(prefer_qpu: bool = False, seed: int | None = None) -> Backend` —
  tier order: QPU (only if `prefer_qpu` and available) → CUDA → native CPU →
  pure-Python `SimulatorBackend`. Never raises. Logs nothing; returns the instance.
- `def tier_report() -> dict` — `{"qpu": bool, "cuda": bool | str(device name), "native_cpu": bool, "python": True}`.

## 7. `PatternRecognizer` hook (minimal edit to `ultraquant/pattern/recognition.py`)

Add optional constructor kwarg `extractor=None`. When set, `train()` and
`evaluate()` compute quantum features via `extractor.extract_batch(all_row_means)`
in one call instead of per-sample `qmap.extract`. Per-sample `features()` /
`recognize()` behavior is unchanged. Keep the edit surgical; all existing tests
must still pass.

## 8. `ultraquant/bench.py`

CLI `python -m ultraquant.bench` (args: `--samples N` default 512, `--qubits N`
default 5, `--big-qubits N` default 20, `--repeat N` default 3):

1. Feature-map batch throughput (samples/s) for: pure Python, C++ 1 thread,
   C++ all threads, CUDA, heterogeneous CPU+GPU split (report the split it chose).
2. Single large-circuit latency at `--big-qubits` (random ry/cnot ring circuit,
   depth 8): pure Python vs C++ vs CUDA.
3. Ternary inference latency (the 30→32→8 demo net shape, batch 512): pure Python
   vs C++ vs CUDA.
Print an aligned table with speedups vs pure Python; skip rows for absent tiers
with a note rather than failing. Exit 0.

## 9. Tests

- `tests/test_native_cpu.py` — `@unittest.skipUnless(cpu tier loads, ...)`:
  parity vs pure Python on 20 random seeded circuits (mixed gates, 2–8 qubits):
  probabilities and expectations within 1e-9; feature-map single + batch parity
  (incl. re-uploading case 7 features / 3 qubits); ternary_forward parity vs
  `TernaryLinear.forward`; batch with `n_threads=4` == `n_threads=1`.
- `tests/test_native_cuda.py` — `@unittest.skipUnless(gpu tier loads, ...)`:
  same parity battery vs pure Python; plus `uqg_run_circuit` at 12 qubits vs
  pure Python (1e-9); `uqg_available() >= 1`.
- `tests/test_native_dispatch.py` — `best_backend()` returns *some* Backend and
  its Bell-state expectations are [0, 0] (±1e-9); `tier_report()` keys; hetero
  extractor parity vs pure Python for each available `force` mode and for the
  auto split; recognizer trained with `extractor=` reaches the same accuracy
  (±0.05) as without on the small classical-check config.
- ALL tests must skip gracefully (not fail) when a tier is absent, so the suite
  stays green on machines with no compiler and no GPU.
