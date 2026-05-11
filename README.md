# 🔥 ETF Hawkes Process – Daily Jump Contagion Engine

**Predict next‑day jump intensity (upside) for ETFs** using multivariate Hawkes processes, then select the best ETF per universe to trade.  
Runs four variants in parallel on your existing `master_data.parquet` (no new data required).

---

## 🧠 Core Concept

Hawkes processes are self‑exciting point processes. When a large return (jump) occurs, the intensity (probability rate) of future jumps temporarily increases – capturing momentum, contagion, and risk clustering.

This engine applies **multivariate Hawkes** to daily log‑returns of ETFs, using detected **signed jumps** (positive and negative large moves). It is designed to:

- Forecast next‑day **upside jump intensity** (proxy for expected return)
- Identify which ETF is most likely to deliver a large positive return tomorrow
- Work across three predefined universes: FI/Commodities, Equity Sectors, and Combined

---

## 📦 Repository Structure
.
├── .github/workflows/daily_run.yml # Daily scheduled execution
├── config.py # Universe definitions, thresholds
├── data_manager.py # Load master_data.parquet from HF
├── hawkes_models.py # Four Hawkes variants
├── trainer.py # Walk‑forward estimation & selection
├── push_results.py # Upload results to HF
├── streamlit_app.py # Interactive dashboard
├── us_calendar.py # Next trading day helper
├── requirements.txt # Python dependencies
└── README.md # This file

text

---

## 🚀 How It Works (Step by Step)

1. **Load data** – from Hugging Face dataset `P2SAMAPA/fi-etf-macro-signal-master-data`.
2. **Define events** – for each ETF, daily signed jumps are defined as:
   - `positive jump`: log‑return > 2× rolling median absolute deviation (MAD)
   - `negative jump`: log‑return < –2× rolling MAD
3. **Walk‑forward estimation** – for each day (starting after 252 days of history):
   - Fit **four Hawkes variants** on the last 252 days:
     - `Exponential` – univariate, positive jumps only.
     - `Signed` – bivariate (pos/neg) with cross‑excitation.
     - `Volatility‑excited` – events based on volatility spikes.
     - `Ensemble` – average of the above three.
   - Predict next‑day positive jump intensity for each variant.
4. **Select the best ETF per universe** – for each ETF, take the **maximum intensity across the four variants**; then pick the ETF with the highest such intensity.
5. **Save results** – a JSON file containing the best ETF, its intensity, and the winning variant.
6. **Upload to Hugging Face** – stored in `P2SAMAPA/p2-etf-hawkes-process-results`.
7. **Display** – Streamlit dashboard loads the latest result and shows a hero box per universe.

---

## 📊 Output Example (Streamlit)

| Universe         | Best ETF | Intensity | Winning Variant |
|------------------|----------|-----------|------------------|
| FI_COMMODITIES   | GLD      | 0.087     | signed          |
| EQUITY_SECTORS   | XLK      | 0.123     | exponential     |
| COMBINED         | TLT      | 0.095     | ensemble        |

> **Interpretation:** Higher intensity → higher expected probability of a large positive jump → stronger buy signal.

---

## ⚙️ Configuration (`config.py`)

Key parameters you can adjust:

- `ROLLING_WINDOW = 252` – number of days used for each fit.
- `JUMP_THRESHOLD_MULTIPLIER = 2.0` – sensitivity of jump detection.
- `VARIANTS = ["exponential", "signed", "volatility", "ensemble"]` – all four run in parallel.
- `TOP_N = 1` – number of top ETFs to display (currently 1 per universe).
