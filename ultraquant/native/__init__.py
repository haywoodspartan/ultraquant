"""UltraQuant native acceleration tier.

ctypes loaders and typed wrappers (:mod:`ultraquant.native.accel`) over the
optional C++ CPU DLL (``_bin/ultraquant_native.dll``) and CUDA GPU DLL
(``_bin/ultraquant_cuda.dll``). Every native path mirrors the pure-Python
tier's semantics (little-endian qubit indexing, SPEC.md feature map) to 1e-9;
when a DLL is absent the loaders return ``None`` and the pure-Python tier
remains the fallback.
"""
