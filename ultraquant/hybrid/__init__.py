"""Hybrid intelligence: quantum and classical experts in one shard library.

The quantum layer earns its place here on a measurable property, not on a
speedup: a variational circuit reaches comparable accuracy with roughly an order
of magnitude fewer parameters than the classical net it replaces. For a system
whose entire thesis is *never load the whole model*, a smaller expert is worth
having on its own terms — it pages faster, more of it stays resident, and more
of the library fits under a given RAM budget.

Which kind of expert wins is decided per category by measurement, and either
kind can lose.
"""

from ultraquant.hybrid.expert import QuantumExpert, quantum_expert_payload
from ultraquant.hybrid.pool import HybridExpertPool, HybridReport

__all__ = [
    "QuantumExpert",
    "quantum_expert_payload",
    "HybridExpertPool",
    "HybridReport",
]
