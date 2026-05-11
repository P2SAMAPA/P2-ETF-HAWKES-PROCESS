"""
Main training script – fast version using exponentially weighted intensity.
Outputs top 3 ETFs per universe by predicted jump intensity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from hawkes_models import FastHawkes, HawkesSigned, HawkesVolatilityExcited, HawkesEnsemble

def detect_jumps(returns, percentile=90):
    """Detect signed jumps using rolling percentile threshold."""
    threshold = returns.abs().rolling(window=252, min_periods=50).quantile(percentile / 100.0)
    threshold.fillna(returns.abs().quantile(percentile / 100.0), inplace=True)
    pos_jumps = (returns > threshold).fillna(False)
    neg_jumps = (returns < -threshold).fillna(False)
    return pos_jumps, neg_jumps

def compute_intensities(returns_series, baseline=0.01, alpha=0.5, decay=0.9):
    """Compute next-day intensity for all four variants."""
    pos_jumps, neg_jumps = detect_jumps(returns_series, percentile=90)

    model_exp = FastHawkes(baseline=baseline, alpha=alpha, decay=decay)
    model_exp.fit(pos_jumps.values)
    exp_intensity = model_exp.predict_next_day_intensity(len(returns_series) - 1, None)

    model_signed = HawkesSigned(baseline=baseline, alpha=alpha, decay=decay)
    model_signed.fit(pos_jumps.values, neg_jumps.values)
    signed_intensity = model_signed.predict_next_day_intensity(len(returns_series) - 1, None)

    model_vol = HawkesVolatilityExcited(baseline=baseline, alpha=alpha, decay=decay)
    model_vol.fit(returns_series)
    vol_intensity = model_vol.predict_next_day_intensity(len(returns_series) - 1)

    model_ens = HawkesEnsemble(baseline=baseline, alpha=alpha, decay=decay)
    model_ens.fit(returns_series, pos_jumps.values, neg_jumps.values)
    ens_intensity = model_ens.predict_next_day_intensity(len(returns_series) - 1, None)

    return {
        "exponential": exp_intensity,
        "signed": signed_intensity,
        "volatility": vol_intensity,
        "ensemble": ens_intensity
    }

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        # Store (ticker, max_intensity, winning_variant) for each ETF
        etf_scores = []

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            series = returns[ticker].dropna()
            if len(series) < config.ROLLING_WINDOW:
                print(f"  {ticker}: insufficient data (< {config.ROLLING_WINDOW} days)")
                continue

            train_series = series.iloc[-config.ROLLING_WINDOW:]
            intensities = compute_intensities(train_series)

            best_intensity = max(intensities.values())
            best_variant = max(intensities, key=intensities.get)

            etf_scores.append({
                "ticker": ticker,
                "intensity": best_intensity,
                "winning_variant": best_variant
            })

        if not etf_scores:
            print(f"  No valid predictions for universe {universe_name}")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort by intensity descending and take top 3
        etf_scores.sort(key=lambda x: x["intensity"], reverse=True)
        top3 = etf_scores[:3]

        print(f"  Top 3 ETFs for {universe_name}:")
        for rank, etf in enumerate(top3, 1):
            print(f"    {rank}. {etf['ticker']} (intensity={etf['intensity']:.6f}, variant={etf['winning_variant']})")

        all_results[universe_name] = {
            "top_etfs": top3,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/hawkes_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    # Upload to Hugging Face
    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Hawkes Process Engine complete (top 3 per universe) ===")

if __name__ == "__main__":
    main()
