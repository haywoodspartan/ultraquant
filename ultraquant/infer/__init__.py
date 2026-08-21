"""Running a converted transformer, rather than only storing one.

§11.96 put tensors into the library, §11.98 got them back out into a
network that computes. Neither built the thing a transformer
actually is. This package is that: the operations between the
matmuls - normalisation, attention, the activation functions - which
`model/network.py` never needed because an MLP is a stack of linear
layers and a ReLU.

Deliberately shared. A vision encoder and a text decoder differ in
their tokenizer, their cache, their mask and their position
encoding; they do NOT differ in what a LayerNorm is or how softmax
attention works. Those live here once.
"""

from __future__ import annotations

__all__ = ["ops"]
