#!/usr/bin/env python3
"""
api.py
======
Short, stable compatibility shim that re-exports the public optlib API.
Prefer importing directly from `optlib`, but this module offers a concise alias.
"""

from optlib.utils.tensor import device, tensor_dtype, complex_dtype  # noqa: F401
from optlib.pricing.bs import bs_price, norm_cdf, norm_pdf            # noqa: F401
from optlib.pricing.iv import implied_vol_from_price                  # noqa: F401
from optlib.models.heston import heston_char_func, bates_char_func    # noqa: F401
from optlib.pricing.cos import cos_price_from_cf                      # noqa: F401
from optlib.sim.paths import generate_heston_paths                    # noqa: F401
from optlib.hedge.delta import (
    delta_hedge_sim,
    compute_per_path_deltas_scaling,
)                                                                     # noqa: F401
from optlib.metrics.performance import calculate_performance_metrics  # noqa: F401

__all__ = [
    'device', 'tensor_dtype', 'complex_dtype',
    'bs_price', 'norm_cdf', 'norm_pdf',
    'implied_vol_from_price',
    'heston_char_func', 'bates_char_func',
    'cos_price_from_cf',
    'generate_heston_paths',
    'delta_hedge_sim', 'compute_per_path_deltas_scaling',
    'calculate_performance_metrics',
]

