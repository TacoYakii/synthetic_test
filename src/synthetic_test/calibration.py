"""Calibration knobs for synthetic base forecasters.

A probabilistic forecast is *calibrated* when its declared spread matches
the magnitude of its actual error, *overconfident* when the declared
spread is narrower than the actual error, and *overdispersed* when it is
wider.

In this harness, each location-scale :class:`ShapeFamily` decomposes the
two into distinct primitives:

* ``bias_std`` — std of the mean-zero Gaussian added to the observed
  value before it becomes the forecast's location. Under this
  construction, the forecast's actual error has std ``≈ bias_std``.
* ``spread`` — the family's declared scale (marginal std for Gaussian,
  scale parameter × ``sqrt(df/(df-2))`` for Student-t).

So the calibration ratio is simply ``spread / bias_std``. The helpers
below turn a categorical mode plus a scalar ``error_scale`` into that
pair, so callers can say "overconfident at error_scale=1.0" without
having to remember which knob controls what.

Non-location-scale families (``gamma``, ``lognormal``, ``zib``) declare
spreads that depend on the observed value's scale, so a scalar
calibration ratio is ill-defined for them; they intentionally do not
expose ``from_calibration`` yet.
"""

from __future__ import annotations

from typing import Literal, Union

CalibrationMode = Literal["calibrated", "overconfident", "overdispersed"]
"""Categorical calibration knob for :func:`ratio`."""

CalibrationSpec = Union[CalibrationMode, float]
"""Either a categorical mode or a raw spread ratio.

Passing a ``float`` bypasses the categorical mode and uses the number
directly as the ``declared_spread / actual_error`` ratio. Values in
``(0, 1)`` yield overconfident forecasts; ``1.0`` is calibrated; values
above ``1`` are overdispersed. This is the direct dial for sweeps.
"""


DEFAULT_OVERCONFIDENT_RATIO = 0.5
DEFAULT_OVERDISPERSED_RATIO = 2.0


def ratio(
    mode: CalibrationSpec,
    overconfident_ratio: float = DEFAULT_OVERCONFIDENT_RATIO,
    overdispersed_ratio: float = DEFAULT_OVERDISPERSED_RATIO,
) -> float:
    """Return the multiplicative ratio ``declared_spread / actual_error`` for ``mode``.

    Args:
        mode: Either a categorical mode string (``"calibrated"``,
            ``"overconfident"``, ``"overdispersed"``) or a positive
            ``float`` used directly as the spread ratio. The float form
            is what you sweep over.
        overconfident_ratio: Ratio used for the ``"overconfident"``
            categorical shortcut; must be strictly in ``(0, 1)``.
            Ignored when ``mode`` is a float.
        overdispersed_ratio: Ratio used for the ``"overdispersed"``
            categorical shortcut; must be strictly greater than ``1``.
            Ignored when ``mode`` is a float.
    """
    if isinstance(mode, bool):
        # bool is a subclass of int; reject it explicitly to avoid True→1.0 surprises.
        raise TypeError(f"mode must be str or float, not bool: {mode!r}")
    if isinstance(mode, (int, float)):
        r = float(mode)
        if r <= 0.0:
            raise ValueError(f"spread ratio must be positive; got {r}")
        return r
    if mode == "calibrated":
        return 1.0
    if mode == "overconfident":
        if not (0.0 < overconfident_ratio < 1.0):
            raise ValueError(
                f"overconfident_ratio must be in (0, 1); got {overconfident_ratio}"
            )
        return overconfident_ratio
    if mode == "overdispersed":
        if not (overdispersed_ratio > 1.0):
            raise ValueError(
                f"overdispersed_ratio must be > 1; got {overdispersed_ratio}"
            )
        return overdispersed_ratio
    raise ValueError(
        f"unknown calibration mode {mode!r}; expected a positive float or "
        "one of 'calibrated', 'overconfident', 'overdispersed'"
    )


__all__ = [
    "CalibrationMode",
    "CalibrationSpec",
    "DEFAULT_OVERCONFIDENT_RATIO",
    "DEFAULT_OVERDISPERSED_RATIO",
    "ratio",
]
