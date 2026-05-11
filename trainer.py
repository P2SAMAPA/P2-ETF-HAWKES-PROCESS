"""
Main training script – fast version using exponentially weighted intensity.
Runs in minutes instead of hours.
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
    """
    Detect signed jumps using rolling percentile threshold.
    Returns boolean series for positive and negative jumps.
    """
    # Rolling absolute return threshold
    threshold = returns.abs().rolling(window=252, min_periods=50).quantile(percentile / 100.0)
    # Fill missing with global quantile
    threshold.fillna(returns.abs().quantile(percentile / 100.0), inplace=True)
    pos_jumps = (returns > threshold).fillna(False)
    neg_jumps = (returns < -threshold).fillna(False)
    return pos_jumps, neg_jumps

def compute_intensities(returns_series, baseline=0.01, alpha=0.5, decay=0.9):
    """
    For a given ETF return series, compute the next-day intensity for all four variants
    using the fast exponential decay method.
    Returns a dict with last (most recent) intensity for each variant.
    """
    # Detect jumps on the full series (walk‑forward inside each variant)
    pos_jumps, neg_jumps = detect_jumps(returns_series, percentile=90)

    # Exponential variant: uses positive jumps
    model_exp = FastHawkes(baseline=baseline, alpha=alpha, decay=decay)
    model_exp.fit(pos_jumps.values)
    exp_intensity = model_exp.predict_next_day_intensity(len(returns_series) - 1, None)

    # Signed variant: uses positive jumps only (same as exponential for upside)
    model_signed = HawkesSigned(baseline=baseline, alpha=alpha, decay=decay)
    model_signed.fit(pos_jumps.values, neg_jumps.values)
    signed_intensity = model_signed.predict_next_day_intensity(len(returns_series) - 1, None)

    # Volatility‑excited variant: detects volatility jumps
    model_vol = HawkesVolatilityExcited(baseline=baseline, alpha=alpha, decay=decay)
    model_vol.fit(returns_series)
    vol_intensity = model_vol.predict_next_day_intensity(len(returns_series) - 1)

    # Ensemble: average of the above three
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

        best_etf = None
        best_intensity = -np.inf
        best_variant = None

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            series = returns[ticker].dropna()
            if len(series) < config.ROLLING_WINDOW:
                print(f"  {ticker}: insufficient data (< {config.ROLLING_WINDOW} days)")
                continue

            # Use only the last ROLLING_WINDOW days for estimation
            train_series = series.iloc[-config.ROLLING_WINDOW:]
            intensities = compute_intensities(train_series)

            # For each ETF, take the maximum intensity across variants
            etf_best = max(intensities.values())
            if etf_best > best_intensity:
                best_intensity = etf_best
                best_etf = ticker
                best_variant = max(intensities, key=intensities.get)

        if best_etf is None:
            print(f"  No valid predictions for universe {universe_name}")
            all_results[universe_name] = {"best_etf": None, "message": "no prediction"}
        else:
            print(f"  Best ETF: {best_etf} with intensity {best_intensity:.6f} (variant: {best_variant})")
            all_results[universe_name] = {
                "best_etf": best_etf,
                "intensity": float(best_intensity),
                "winning_variant": best_variant,
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
    print("\n=== Hawkes Process Engine complete (fast version) ===")

if __name__ == "__main__":
    main()
