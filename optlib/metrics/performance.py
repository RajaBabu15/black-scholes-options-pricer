# Performance metrics wrapper
try:
    from options_engine_torch import calculate_performance_metrics
except Exception:
    def calculate_performance_metrics(*args, **kwargs):
        raise NotImplementedError("calculate_performance_metrics not available")

