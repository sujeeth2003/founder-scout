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

    st.markdown("**Founder Signals**")
    c1, c2 = st.columns(2)
    n_founders          = c1.number_input("# Founders", 1, 6, 2)
    prior_exits         = c2.number_input("Prior exits", 0, 4, 0)
    founder_ivy         = c1.checkbox("Ivy / top-10 university")
    founder_phd         = c2.checkbox("PhD founder")

    st.markdown("**Funding & Team**")
    c3, c4 = st.columns(2)
    team_size      = c3.number_input("Team size at seed", 1, 20, 3)
    seed_amount    = c4.number_input("Seed amount ($)", 100_000, 5_000_000, 500_000, step=50_000)
    months_since   = c3.slider("Months since seed", 1, 48, 10)
    pivot_count    = c4.number_input("# Pivots", 0, 4, 0)

    st.markdown("**Traction**")
    c5, c6 = st.columns(2)
    gh_stars    = c5.number_input("GitHub stars (proxy)", 0, 10_000, 300)
    press       = c6.number_input("Press mentions", 0, 50, 3)
    has_patent  = st.checkbox("Has patent")
    domain_fit  = st.slider("Domain overlap with fund thesis", 0.0, 1.0, 0.7)

    run = st.button("Score & Generate Memo", type="primary")

# ─── Output ───────────────────────────────────────────────────────────────────
with col2:
    st.subheader("Output")

    if run:
        profile = {
            "company":             company,
            "sector":              sector,
            "description":         description,
            "n_founders":          int(n_founders),
            "founder_prior_exits": int(prior_exits),
            "founder_ivy":         int(founder_ivy),
            "founder_phd":         int(founder_phd),
            "team_size_at_seed":   int(team_size),
            "seed_amount_usd":     int(seed_amount),
            "months_since_seed":   int(months_since),
            "pivot_count":         int(pivot_count),
            "has_patent":          int(has_patent),
            "github_stars_proxy":  int(gh_stars),
            "press_mentions":      int(press),
            "domain_overlap":      float(domain_fit),
        }
        if gh_user:
            profile["github_username"] = gh_user

        with st.spinner("Running scoring pipeline..."):
            try:
                result = generate_memo(profile)
                score  = result["model_score"]
                memo   = result["memo_markdown"]

                prob  = score["series_a_probability"]
                label = score["score_label"]
                color = "#2ecc71" if prob > 0.65 else "#f39c12" if prob > 0.40 else "#e74c3c"

                st.markdown(f"""
                <div style='background:{color}22; border-left:4px solid {color};
                            padding:12px; border-radius:6px; margin-bottom:12px'>
                    <b style='font-size:1.4em'>{prob:.0%}</b> Series A Probability
                    &nbsp;·&nbsp; <b>{label}</b>
                </div>
                """, unsafe_allow_html=True)

                if score["risk_flags"]:
                    st.warning("**Risk Flags:** " + " · ".join(score["risk_flags"]))

                st.markdown("---")
                st.markdown(memo)

            except FileNotFoundError:
                st.error("Model not trained. Click 'Train / Retrain Model' in the sidebar first.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Fill in the profile on the left and click **Score & Generate Memo**.")
