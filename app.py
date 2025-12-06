"""
app.py
------
Streamlit demo — paste a startup profile, get a scored investment memo.

Run: streamlit run app.py
"""

import sys
import json
import streamlit as st

sys.path.insert(0, "src")
from src.model import load_model, score_startup, train
from src.memo_agent import generate_memo

st.set_page_config(page_title="FounderScout", page_icon="🔭", layout="wide")

st.title("🔭 FounderScout")
st.caption("AI-powered startup scoring & investment memo generation")

# ─── Sidebar: model status ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    if st.button("Train / Retrain Model"):
        with st.spinner("Training on synthetic dataset..."):
            from ingest import generate_synthetic_startups
            df = generate_synthetic_startups(800)
            df.to_csv("data/startups.csv", index=False)
            model, le, auc = train()
        st.success(f"Trained — AUC: {auc:.3f}")

    st.markdown("---")
    st.caption("Optionally set GITHUB_TOKEN env var for live GitHub enrichment")

# ─── Main form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Startup Profile")

    company     = st.text_input("Company Name", "ArcLight AI")
    sector      = st.selectbox("Sector", ["AI/ML", "FinTech", "HealthTech", "DevTools", "Climate", "SaaS", "Robotics"])
    description = st.text_area("One-line description", "LLM observability tooling for enterprise AI teams")
    gh_user     = st.text_input("GitHub username (optional — pulls live data)", "")

