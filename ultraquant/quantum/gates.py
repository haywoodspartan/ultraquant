"""Standard quantum gate matrices as plain nested lists of complex numbers.

All gates are 2x2 matrices in the computational basis. ``Matrix`` is the
exported type alias used throughout the quantum package.
"""

from __future__ import annotations

import cmath
import math

Matrix = list[list[complex]]
"""Type alias for a gate matrix: ``list[list[complex]]``."""

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)

H: Matrix = [
    [complex(_INV_SQRT2), complex(_INV_SQRT2)],
    [complex(_INV_SQRT2), complex(-_INV_SQRT2)],
]
"""Hadamard gate."""

X: Matrix = [
    [0j, 1 + 0j],
    [1 + 0j, 0j],
]
"""Pauli-X (NOT) gate."""

Y: Matrix = [
    [0j, -1j],
    [1j, 0j],
]
"""Pauli-Y gate."""

Z: Matrix = [
    [1 + 0j, 0j],
    [0j, -1 + 0j],
]
"""Pauli-Z gate."""

S: Matrix = [
    [1 + 0j, 0j],
    [0j, 1j],
]
"""Phase gate (sqrt of Z)."""

T: Matrix = [
    [1 + 0j, 0j],
    [0j, cmath.exp(1j * math.pi / 4.0)],
]
"""T gate (pi/8 gate, sqrt of S)."""

I2: Matrix = [
    [1 + 0j, 0j],
    [0j, 1 + 0j],
]
"""2x2 identity."""


def rx(theta: float) -> Matrix:
    """Rotation about the X axis by ``theta`` radians.

    ``[[cos(t/2), -1j*sin(t/2)], [-1j*sin(t/2), cos(t/2)]]``
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return [
        [complex(c), -1j * s],
        [-1j * s, complex(c)],
    ]


def ry(theta: float) -> Matrix:
    """Rotation about the Y axis by ``theta`` radians.

    ``[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]``
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return [
        [complex(c), complex(-s)],
        [complex(s), complex(c)],
    ]


def rz(theta: float) -> Matrix:
    """Rotation about the Z axis by ``theta`` radians.

    ``[[exp(-1j*t/2), 0], [0, exp(1j*t/2)]]``
    """
    return [
        [cmath.exp(-1j * theta / 2.0), 0j],
        [0j, cmath.exp(1j * theta / 2.0)],
    ]
