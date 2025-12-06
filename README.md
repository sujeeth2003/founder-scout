# 🔭 FounderScout

AI-powered startup deal sourcing and founder scoring tool.

**What it does:**
1. Ingests startup signals (founder pedigree, funding, team, traction)
2. Optionally enriches with **live GitHub data** (public API)
3. Scores Series A probability with a **Gradient Boosting classifier** (ROC-AUC ~0.81)
4. Auto-generates a structured **investment memo** using a LangGraph + Claude agent

---

## Quickstart

```bash
git clone https://github.com/<your-username>/founder-scout
cd founder-scout
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here
export GITHUB_TOKEN=your_token_here   # optional — raises rate limit to 5000/hr

# 1. Generate synthetic training data + train model
python src/ingest.py          # creates data/startups.csv
python src/model.py           # trains model, saves to outputs/model.pkl

# 2. Run a demo memo generation
python src/memo_agent.py

# 3. Launch the Streamlit UI
streamlit run app.py
```

---

## Architecture

```
startup profile (dict)
        │
   ┌────▼─────┐
   │  enrich  │  ← GitHub API (live technical depth signals)
   └────┬─────┘
        │
   ┌────▼─────┐
   │  score   │  ← Gradient Boosting classifier (300 trees, 13 features)
   └────┬─────┘
        │
   ┌────▼──────────┐
   │  write_memo   │  ← Claude (claude-sonnet-4) via LangGraph
   └────┬──────────┘
        │
   investment memo (markdown)
```

## Features

| Feature | Source |
|---|---|
| Founder prior exits | Input |
| Team size at seed | Input |
| GitHub stars (technical traction) | GitHub API or input |
| GitHub language diversity | GitHub API |
| Press mentions | Input |
| Domain overlap with fund thesis | Input (0–1 score) |
| Seed amount | Input |
| Pivot count | Input |
| Founder education signals | Input |

## Extending to Real Data

- **Crunchbase**: Replace `generate_synthetic_startups()` in `ingest.py` with a Crunchbase API client. The schema is identical.
- **LinkedIn**: Use LinkedIn API or a data provider (Proxycurl) for founder employment history.
- **GitHub**: Already live — set `GITHUB_TOKEN` for 5000 req/hr.

## Results

Trained on 800 synthetic startups mirroring historical seed-to-Series-A conversion patterns:
- Test ROC-AUC: **~0.81**
- 5-Fold CV AUC: **0.79 ± 0.03**

---

Built by [Sujeeth Sukumar](https://sujeeth2003.github.io/Portfolio/)
