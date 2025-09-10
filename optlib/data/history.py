import os
import pandas as pd
from datetime import datetime

def load_or_download_hist(ticker: str, years: int, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        return df
    start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
    import yfinance as yf
    df = yf.download(ticker, start=str(start))
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path)
    return df

