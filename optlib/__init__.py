# Unified public API for the options library
# Modules re-export for convenience

from .data.market import fetch_risk_free_rate, fetch_dividend_yield
from .data.options import choose_expiries, fetch_clean_chain, load_or_download_chain_clean
from .models.heston import heston_char_func
from .pricing.cos import cos_price_from_cf
from .pricing.iv import implied_vol_from_price
from .sim.paths import generate_heston_paths
from .hedge.delta import delta_hedge_sim, compute_per_path_deltas_scaling
from .metrics.performance import calculate_performance_metrics
from .utils.tensor import tensor_dtype

