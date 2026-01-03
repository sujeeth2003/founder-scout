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
git clone https://github.com/sujeeth2003/founder-scout
cd founder-scout
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here
export GITHUB_TOKEN=your_token_here   # optional — raises rate limit to 5000/hr

# 1. Generate synthetic training data + train model
python src/ingest.py          # creates data/startups.csv
python src/model.py           # trains model, saves to outputs/model.pkl

