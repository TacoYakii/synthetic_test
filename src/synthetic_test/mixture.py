"""Mixture spec — how shape families are distributed across hierarchy nodes.

A :class:`MixtureSpec` is a probability distribution over
:class:`~synthetic_test.families.base.ShapeFamily` instances. At data
generation time each hierarchy node is assigned exactly one family sampled
i.i.d. from this distribution.

Users can supply arbitrary K-way mixtures — this is the primary knob that
drives cross-node dispersion diversity and hence the budget ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from synthetic_test.families.base import ShapeFamily


@dataclass
class MixtureSpec:
    """K-way mixture over shape families with arbitrary weights.

    Attributes:
        components: Sequence of ``(family, weight)`` pairs. Weights need
            not sum to 1 — they are normalized on access.
        label: Optional short human-readable name for logging / plot axes.
    """

    components: list[tuple[ShapeFamily, float]]
    label: str = ""

    def __post_init__(self):
        if len(self.components) == 0:
            raise ValueError("MixtureSpec must have at least one component")
        w = np.asarray([c[1] for c in self.components], dtype=float)
        if (w < 0).any():
            raise ValueError(f"weights must be non-negative; got {w.tolist()}")
        if w.sum() <= 0:
            raise ValueError("weights must sum to a positive number")

    @property
    def families(self) -> list[ShapeFamily]:
        """The K family instances, in the order given."""
        return [c[0] for c in self.components]

    @property
    def weights(self) -> np.ndarray:
        """Normalized weight vector of length K."""
        w = np.asarray([c[1] for c in self.components], dtype=float)
        return w / w.sum()

    def assign(self, num_nodes: int, rng: np.random.Generator) -> list[ShapeFamily]:
        """Return per-node family assignment by i.i.d. categorical draw."""
        idx = rng.choice(len(self.components), size=num_nodes, p=self.weights)
        return [self.components[i][0] for i in idx]

    def describe(self) -> dict:
        """Serializable summary for logging."""
        return {
            "label": self.label,
            "components": [
                {"family": fam.name, "weight": float(w), "params": _family_params(fam)}
                for fam, w in zip(self.families, self.weights)
            ],
        }


def _family_params(family: ShapeFamily) -> dict:
    """Extract the dataclass fields of a family instance for logging."""
    from dataclasses import fields

    return {
        f.name: getattr(family, f.name)
        for f in fields(family)
        if f.name != "name"
    }


def uniform_mixture(families: Iterable[ShapeFamily], label: str = "") -> MixtureSpec:
    """Convenience: build a MixtureSpec with equal weights over ``families``."""
    fams = list(families)
    return MixtureSpec(components=[(f, 1.0) for f in fams], label=label)


def dirichlet_mixture(
    families: Sequence[ShapeFamily],
    alpha: float | Sequence[float],
    rng: np.random.Generator,
    label: str = "",
) -> MixtureSpec:
    """Draw a random weight vector from a Dirichlet(alpha) and wrap it."""
    K = len(families)
    if np.isscalar(alpha):
        alpha_vec = np.full(K, float(alpha))
    else:
        alpha_vec = np.asarray(alpha, dtype=float)
        if alpha_vec.shape != (K,):
            raise ValueError(f"alpha length must equal len(families)={K}")
    w = rng.dirichlet(alpha_vec)
    return MixtureSpec(components=list(zip(families, w)), label=label)


__all__ = ["MixtureSpec", "uniform_mixture", "dirichlet_mixture"]
