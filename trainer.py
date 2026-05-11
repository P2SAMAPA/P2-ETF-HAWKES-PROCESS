"""
Main training script: for each universe, for each ETF, fit all four Hawkes variants,
compute predicted positive jump intensity for next day (proxy for expected return),
select the best variant (highest intensity) for each ETF, then pick the ETF with highest
intensity across variants (top 1) per universe.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from hawkes_models import HawkesExponential, HawkesSigned, HawkesVolatilityExcited, HawkesEnsemble

def detect_jumps(returns, threshold_mult=2.0):
    """Detect signed jumps: large absolute returns beyond rolling MAD."""
    mad = returns.rolling(window=252).apply(lambda x: np.median(np.abs(x - np.median(x))) * 1.4826, raw=False)
    threshold = threshold_mult * mad
    pos_jumps = (returns > threshold).fillna(False)
    neg_jumps = (returns < -threshold).fillna(False)
    return pos_jumps, neg_jumps

def rolling_hawkes_predictions(returns_series, variant_names, window=252):
    """
    Walk-forward: for each day t >= window, fit on returns[t-window:t] and predict next day.
    Returns a DataFrame of predicted intensities for each variant.
    """
    dates = returns_series.index
    n = len(dates)
    preds = pd.DataFrame(index=dates, columns=variant_names)
    for i in range(window, n):
        train_returns = returns_series.iloc[i-window:i]
        # Detect jumps in training window
        pos_jumps, neg_jumps = detect_jumps(train_returns, threshold_mult=config.JUMP_THRESHOLD_MULTIPLIER)
        # Fit each variant
        model_exp = HawkesExponential(decay=1.0).fit(pos_jumps.values)
        model_signed = HawkesSigned(decay=1.0).fit(pos_jumps.values, neg_jumps.values)
        model_vol = HawkesVolatilityExcited(decay=1.0).fit(train_returns)
        model_ens = HawkesEnsemble(decay=1.0).fit(train_returns, pos_jumps.values, neg_jumps.values)
        # Predict next day intensity
        last_idx = window - 1  # index within train window
        preds.loc[dates[i], "exponential"] = model_exp.predict_next_day_intensity(last_idx, window)
        preds.loc[dates[i], "signed"] = model_signed.predict_next_day_intensity(last_idx, window)
        preds.loc[dates[i], "volatility"] = model_vol.predict_next_day_intensity(last_idx)
        preds.loc[dates[i], "ensemble"] = model_ens.predict_next_day_intensity(last_idx, window)
    return preds

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
            if len(series) < config.ROLLING_WINDOW + 10:
                print(f"  {ticker}: insufficient data")
                continue

            # Get predictions (intensity) for each variant
            preds = rolling_hawkes_predictions(series, config.VARIANTS, window=config.ROLLING_WINDOW)
            # Use the latest prediction (last row) as next-day intensity
            latest = preds.iloc[-1] if not preds.empty else None
            if latest is None:
                continue
            # For each ETF, take the max intensity across variants (or average? we take max as "best outcome")
            etf_best_intensity = latest.max()
            if etf_best_intensity > best_intensity:
                best_intensity = etf_best_intensity
                best_etf = ticker
                best_variant = latest.idxmax()

        if best_etf is None:
            print(f"  No valid predictions for universe {universe_name}")
            all_results[universe_name] = {"best_etf": None, "message": "no prediction"}
        else:
            print(f"  Best ETF: {best_etf} with intensity {best_intensity:.4f} (variant: {best_variant})")
            all_results[universe_name] = {
                "best_etf": best_etf,
                "intensity": float(best_intensity),
                "winning_variant": best_variant,
                "run_date": today
            }

    # Save results to JSON
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/hawkes_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    # Push to Hugging Face (implement push_results.py)
    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Hawkes Process Engine complete ===")

if __name__ == "__main__":
    main()
