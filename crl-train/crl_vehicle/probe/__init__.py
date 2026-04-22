"""Probe-time utilities: prior recalibration for classifier heads."""
from crl_vehicle.probe.recalibration import (
    compute_binary_prior,
    compute_multiclass_prior,
    apply_binary_log_prior_shift,
    apply_multiclass_log_prior_shift,
    UNIFORM_BINARY_PRIOR,
    uniform_multiclass_prior,
)

__all__ = [
    "compute_binary_prior",
    "compute_multiclass_prior",
    "apply_binary_log_prior_shift",
    "apply_multiclass_log_prior_shift",
    "UNIFORM_BINARY_PRIOR",
    "uniform_multiclass_prior",
]
