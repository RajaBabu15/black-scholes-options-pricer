# Hedging and per-path delta wrappers
try:
    from options_engine_torch import delta_hedge_sim, compute_per_path_deltas_scaling
except Exception:
    def delta_hedge_sim(*args, **kwargs):
        raise NotImplementedError("delta_hedge_sim not available")
    def compute_per_path_deltas_scaling(*args, **kwargs):
        raise NotImplementedError("compute_per_path_deltas_scaling not available")

