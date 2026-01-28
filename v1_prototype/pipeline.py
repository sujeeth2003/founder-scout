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


def load_startup_data(csv_path: str) -> pd.DataFrame:
    """
    Load Crunchbase-style CSV.
    Expected columns: founder_name, prior_exits, team_size, domain,
    fund_thesis_overlap (0-1), raised_seed, series_a_success (label).
    A sample CSV is in data/sample_startups.csv
    """
    return pd.read_csv(csv_path)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from raw startup + founder fields.

    Features:
    - prior_exits: how many companies this founder has exited before
    - github_stars_log: log-scaled GitHub stars (technical depth proxy)
    - has_ml_repos: binary flag for ML/data work
    - team_size: founding team headcount
    - domain_match: does domain overlap with fund thesis (0-1 float)
    - raised_seed: already raised seed? binary
    - founder_follower_log: GitHub followers log-scaled
    """
    df = df.copy()

    # Log-scale skewed counts to prevent GB from over-indexing on outliers
    df["github_stars_log"] = np.log1p(df.get("total_stars", 0))
    df["founder_follower_log"] = np.log1p(df.get("followers", 0))

    # Composite "technical depth" score
    df["tech_depth_score"] = (
        df["github_stars_log"] * 0.4 +
        df["has_ml_repos"] * 0.3 +
        df["num_languages"].clip(0, 10) / 10 * 0.3
    )

    # Binary signals
    df["raised_seed"] = df["raised_seed"].astype(int)
    df["has_prior_exit"] = (df["prior_exits"] > 0).astype(int)

    feature_cols = [
        "prior_exits", "has_prior_exit",
        "tech_depth_score", "github_stars_log", "has_ml_repos", "founder_follower_log",
        "team_size", "fund_thesis_overlap", "raised_seed", "account_age_years"
    ]

    return df[feature_cols + ["series_a_success"]]


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train Gradient Boosting classifier on engineered features.
    Returns trained model + scaler + feature list + evaluation dict.
    """
    feature_cols = [c for c in df.columns if c != "series_a_success"]
    X = df[feature_cols].values
    y = df["series_a_success"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,          # shallow trees → less overfitting on small data
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✅ ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature Importances:")
    print(importance.sort_values(ascending=False).to_string())

    return model, scaler, feature_cols, {"roc_auc": auc}


def save_model(model, scaler, feature_cols, path="models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f"{path}/gb_founder_model.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")
    with open(f"{path}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"Model saved to {path}/")


def load_model(path="models/"):
    model = joblib.load(f"{path}/gb_founder_model.pkl")
    scaler = joblib.load(f"{path}/scaler.pkl")
    with open(f"{path}/feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


# ─────────────────────────────────────────────
# 4. SCORING A NEW FOUNDER
# ─────────────────────────────────────────────

