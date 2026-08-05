"""Reasoning surfaces: structures where more than one result can be held at once.

The interpreter's original reasoning step mapped one input to one expert to one
label, which has no room for a partial answer. This package holds the structures
that do.
"""

from ultraquant.reason.blackboard import (
    Blackboard,
    ConstraintContributor,
    Contribution,
    Contributor,
    ExpertContributor,
    compose,
    run_blackboard,
)

__all__ = [
    "Blackboard",
    "Contribution",
    "Contributor",
    "ExpertContributor",
    "ConstraintContributor",
    "run_blackboard",
    "compose",
]
