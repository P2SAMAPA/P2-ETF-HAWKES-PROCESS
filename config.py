"""
Configuration for Hawkes Process Jump Contagion Engine.
"""

# Hugging Face settings
HF_TOKEN = "your_hf_token_here"  # Set via secrets in GitHub Actions
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-hawkes-process-results"

# Universe definitions (same as before)
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Hawkes parameters
ROLLING_WINDOW = 252          # days
REFIT_DAILY = True            # refit every day (walk-forward)
JUMP_THRESHOLD_MULTIPLIER = 2.0   # 2 × rolling MAD for large returns
JUMP_PERCENTILE = 95.0        # alternative: 95th percentile

# Variants to run in parallel (all four)
VARIANTS = ["exponential", "signed", "volatility", "ensemble"]

# Kernel choices
KERNEL = "exponential"        # for exponential variant: "exponential" or "powerlaw"

# Volatility‑excited: use Parkinson volatility (or realized)
USE_PARKINSON_VOL = True

# Number of days ahead for intensity forecast
FORECAST_DAYS = 1

# Output
TOP_N = 1                     # display top ETF per universe
