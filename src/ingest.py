"""
ingest.py
---------
Pulls founder + startup signals from GitHub API (live) and a local
synthetic dataset that mirrors Crunchbase schema.

To extend to real Crunchbase: swap generate_synthetic_startups() with
a Crunchbase API client — the schema is identical.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ─── GitHub API ───────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # optional — raises rate limit from 60 to 5000 req/hr

def github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def fetch_github_founder(username: str) -> dict:
    """
    Pull technical depth signals for a founder from GitHub.
    Returns a flat dict of features.
    """
    base = "https://api.github.com"
    headers = github_headers()

    # User profile
    r = requests.get(f"{base}/users/{username}", headers=headers, timeout=10)
    if r.status_code != 200:
        return {"github_username": username, "github_fetch_ok": False}
    u = r.json()

    # Repo list (first 100)
    repos_r = requests.get(
        f"{base}/users/{username}/repos?per_page=100&sort=pushed",
        headers=headers, timeout=10
    )
    repos = repos_r.json() if repos_r.status_code == 200 else []
    time.sleep(0.5)  # stay polite

    total_stars    = sum(r.get("stargazers_count", 0) for r in repos)
    total_forks    = sum(r.get("forks_count", 0) for r in repos)
    repo_count     = len(repos)
    languages      = list({r.get("language") for r in repos if r.get("language")})
    has_ml_repo    = any(
        kw in (r.get("description") or "").lower() or
        kw in (r.get("name") or "").lower()
        for r in repos for kw in ["ml", "ai", "model", "neural", "gpt", "llm", "predict"]
    )

    account_age_days = 0
    if u.get("created_at"):
        created = datetime.fromisoformat(u["created_at"].replace("Z", "+00:00"))
        account_age_days = (datetime.now(timezone.utc) - created).days

    return {
        "github_username":    username,
        "github_fetch_ok":    True,
        "gh_followers":       u.get("followers", 0),
        "gh_public_repos":    repo_count,
        "gh_total_stars":     total_stars,
        "gh_total_forks":     total_forks,
        "gh_languages_count": len(languages),
        "gh_has_ml_repo":     int(has_ml_repo),
        "gh_account_age_days":account_age_days,
        "gh_hireable":        int(bool(u.get("hireable"))),
        "gh_blog":            int(bool(u.get("blog"))),
    }


# ─── Synthetic Crunchbase-style dataset ───────────────────────────────────────

def generate_synthetic_startups(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic startup dataset.
    Schema mirrors Crunchbase export format so swapping in real data
    requires only replacing this function with an API call.

    Target label: series_a_success
      1 = raised Series A within 3 years of seed
      0 = did not raise / shut down
    """
    rng = np.random.default_rng(seed)

    sectors = ["AI/ML", "FinTech", "HealthTech", "DevTools", "Climate", "SaaS", "Robotics"]
    locations = ["San Francisco", "New York", "Austin", "Boston", "Seattle", "Remote"]

    n_founders           = rng.integers(1, 5, n)
    founder_prior_exits  = rng.integers(0, 3, n)           # 0–2 prior exits
    founder_ivy          = rng.binomial(1, 0.25, n)        # 25% ivy-league
    founder_phd          = rng.binomial(1, 0.18, n)
    team_size_at_seed    = rng.integers(2, 12, n)
    sector               = rng.choice(sectors, n)
    location             = rng.choice(locations, n)
    seed_amount_usd      = rng.integers(200_000, 3_000_000, n)
    months_since_seed    = rng.integers(6, 48, n)
    pivot_count          = rng.integers(0, 3, n)
    has_patent           = rng.binomial(1, 0.15, n)
    github_stars_proxy   = rng.integers(0, 4000, n)        # proxy for technical traction
    press_mentions       = rng.integers(0, 30, n)
    domain_overlap       = rng.uniform(0, 1, n)            # 0–1 fit with fund thesis

