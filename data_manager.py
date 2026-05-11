"""
Data management: load master data from Hugging Face, compute returns.
"""
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    """Load master_data.parquet from Hugging Face."""
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, tickers):
    """Extract log returns for given tickers."""
    returns = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            # Use adjusted closing prices? We assume 'close' column exists.
            # In master_data.parquet, typically 'close' per ticker.
            price_series = df[ticker] if ticker in df.columns else None
            if price_series is not None and not price_series.isna().all():
                returns[ticker] = np.log(price_series / price_series.shift(1))
    return returns.dropna()
