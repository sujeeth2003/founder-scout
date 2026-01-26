"""
Founder Scoring Pipeline
========================
Ingests startup signals → engineers founder features → trains GB classifier → scores founders.

Data sources used here: GitHub API (free), synthetic Crunchbase-style CSV.
Swap in real Crunchbase/LinkedIn data when you have API access.
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # set in .env


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def fetch_github_profile(username: str) -> dict:
    """
    Pull public GitHub signals for a founder.
    Proxies technical depth without needing LinkedIn API.
    """
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    base = "https://api.github.com"

    try:
        user = requests.get(f"{base}/users/{username}", headers=headers).json()
        repos = requests.get(f"{base}/users/{username}/repos?per_page=100", headers=headers).json()

        if not isinstance(repos, list):
            repos = []

        total_stars = sum(r.get("stargazers_count", 0) for r in repos)
        languages = set(r.get("language") for r in repos if r.get("language"))
        has_ml = any(lang in languages for lang in ["Python", "Jupyter Notebook", "R"])

        return {
            "github_username": username,
            "public_repos": user.get("public_repos", 0),
            "followers": user.get("followers", 0),
            "total_stars": total_stars,
            "num_languages": len(languages),
            "has_ml_repos": int(has_ml),
            "account_age_years": _account_age(user.get("created_at", "")),
        }
    except Exception as e:
        print(f"GitHub fetch failed for {username}: {e}")
        return _empty_github()


def _account_age(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        return (datetime.utcnow() - created).days / 365.25
    except:
        return 0.0


def _empty_github() -> dict:
    return {
        "github_username": "",
        "public_repos": 0, "followers": 0, "total_stars": 0,
        "num_languages": 0, "has_ml_repos": 0, "account_age_years": 0.0
    }


