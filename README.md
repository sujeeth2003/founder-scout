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

