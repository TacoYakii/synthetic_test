"""Registry of shape families.

Built-in families are pre-registered. Users can register new ones with the
:func:`register` decorator; anything conforming to
:class:`~synthetic_test.families.base.ShapeFamily` is accepted.
"""

from __future__ import annotations

from typing import Callable

from synthetic_test.families.base import ShapeFamily
from synthetic_test.families.gamma import GammaForecaster
from synthetic_test.families.gaussian import GaussianForecaster
from synthetic_test.families.lognormal import LogNormalForecaster
from synthetic_test.families.student_t import StudentTForecaster
from synthetic_test.families.zib import ZIBForecaster

REGISTRY: dict[str, type[ShapeFamily]] = {}


def register(name: str) -> Callable[[type[ShapeFamily]], type[ShapeFamily]]:
    """Class decorator to register a new shape family under ``name``."""

    def _wrap(cls: type[ShapeFamily]) -> type[ShapeFamily]:
        if name in REGISTRY:
            raise ValueError(f"family {name!r} already registered")
        REGISTRY[name] = cls
        return cls

    return _wrap


def get(name: str) -> type[ShapeFamily]:
    """Return the family class registered under ``name``."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown family {name!r}; registered: {sorted(REGISTRY)}"
        ) from exc


def available() -> list[str]:
    """Return the sorted list of registered family names."""
    return sorted(REGISTRY)


# Register built-ins.
REGISTRY["gaussian"] = GaussianForecaster
REGISTRY["student_t"] = StudentTForecaster
REGISTRY["lognormal"] = LogNormalForecaster
REGISTRY["gamma"] = GammaForecaster
REGISTRY["zib"] = ZIBForecaster


__all__ = [
    "REGISTRY",
    "ShapeFamily",
    "GaussianForecaster",
    "StudentTForecaster",
    "LogNormalForecaster",
    "GammaForecaster",
    "ZIBForecaster",
    "register",
    "get",
    "available",
]
