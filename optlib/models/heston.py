# Heston model characteristic function re-export
try:
    from options_engine_torch import heston_char_func
except Exception:
    def heston_char_func(*args, **kwargs):
        raise NotImplementedError("heston_char_func not available")

