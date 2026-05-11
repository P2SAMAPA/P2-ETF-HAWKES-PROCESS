"""
Streamlit dashboard showing hero box per universe with best ETF to trade.
"""
import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Hawkes Process Jump Contagion", layout="wide")
st.title("🔥 Hawkes Process: Daily Jump Contagion Engine")
st.caption("Predicts next-day jump intensity (upside). Trades the ETF with highest intensity per universe.")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'hawkes_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No Hawkes results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Multivariate Hawkes process (exponential, signed, volatility-excited, ensemble)")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

st.header("📈 Best ETF to Trade Tomorrow (highest predicted upside intensity)")

cols = st.columns(len(universes))
for idx, (universe_name, uni_data) in enumerate(universes.items()):
    with cols[idx]:
        st.subheader(f"🌍 {universe_name}")
        best_etf = uni_data.get("best_etf", "N/A")
        intensity = uni_data.get("intensity", 0.0)
        variant = uni_data.get("winning_variant", "N/A")
        st.metric(f"🏆 {best_etf}", f"intensity = {intensity:.4f}", f"via {variant}")
        st.caption("Higher intensity → higher expected jump return")

st.divider()
st.caption("Intensity reflects the estimated conditional jump probability per day (positive jumps). The engine runs four Hawkes variants in parallel and selects the ETF with the highest predicted intensity across all variants for each universe.")
