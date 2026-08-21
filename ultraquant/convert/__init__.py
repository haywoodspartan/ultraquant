"""Read a trained transformer, and answer what converting it costs.

The kit is deliberately two things rather than one. Reading a
checkpoint is engineering: formats, dtypes, offsets, and a refusal by
name for what is not supported. Converting one into UltraQuant's
ternary tier is a CLAIM about fidelity, and this project does not
ship claims without measuring them - so the conversion path reports
what it destroyed, per tensor, in the units that matter.
"""

from __future__ import annotations

__all__ = ["gguf", "safetensors"]
