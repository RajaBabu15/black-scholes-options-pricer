# Pricing COS method re-export
try:
    from options_engine_torch import cos_price_from_cf
except Exception:
    def cos_price_from_cf(*args, **kwargs):
        raise NotImplementedError("cos_price_from_cf not available")

