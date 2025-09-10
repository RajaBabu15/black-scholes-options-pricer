# Implied volatility helper re-export
try:
    from options_engine_torch import implied_vol_from_price
except Exception:
    def implied_vol_from_price(*args, **kwargs):
        raise NotImplementedError("implied_vol_from_price not available")

