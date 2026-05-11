"""
Streamlit dashboard showing top 3 ETFs per universe with highest jump intensity.
"""
import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Hawkes Process – Top 3 ETFs", layout="wide")
st.title("🔥 Hawkes Process: Daily Jump Contagion Engine")
st.caption("Predicts next-day jump intensity (upside). Displays top 3 ETFs per universe.")

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
st.sidebar.write("**Method:** Fast Hawkes proxy (exponential decay)")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

st.header("📈 Top 3 ETFs to Trade Tomorrow (by predicted upside intensity)")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.warning(f"No top ETFs for {universe_name}")
        continue

    st.subheader(f"🌍 {universe_name}")
    
    # Create 3 columns for the top 3
    cols = st.columns(3)
    for idx, etf_info in enumerate(top_etfs):
        with cols[idx]:
            ticker = etf_info["ticker"]
            intensity = etf_info["intensity"]
            variant = etf_info["winning_variant"]
            st.metric(
                label=f"#{idx+1} {ticker}",
                value=f"intensity = {intensity:.4f}",
                delta=f"via {variant}"
            )
    st.divider()

# Optional detailed table
with st.expander("📊 Full ranking table"):
    for universe_name, uni_data in universes.items():
        st.write(f"**{universe_name}**")
        top_etfs = uni_data.get("top_etfs", [])
        df = pd.DataFrame(top_etfs)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.write("---")

st.caption("Higher intensity → higher expected probability of a large positive jump. The engine uses four variants (exponential, signed, volatility, ensemble) and picks the best intensity per ETF.")
