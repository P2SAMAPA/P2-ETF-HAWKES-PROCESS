"""
Data management: load master data from Hugging Face, compute returns.
Assumes the parquet file has a datetime index (no separate 'date' column).
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    """Load master_data.parquet from Hugging Face."""
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    # The index is already datetime; ensure it's named 'date' for consistency
    if df.index.name != 'date':
        df.index.name = 'date'
    # If there is a column named 'date', set it as index and drop
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, tickers):
    """Extract log returns for given tickers."""
    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    returns = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            price_series = df[ticker]
            if not price_series.isna().all():
                returns[ticker] = np.log(price_series / price_series.shift(1))
    return returns.dropna()
