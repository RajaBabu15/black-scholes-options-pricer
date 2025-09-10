# Path generation wrappers
try:
    from options_engine_torch import generate_heston_paths
except Exception:
    def generate_heston_paths(*args, **kwargs):
        raise NotImplementedError("generate_heston_paths not available")

