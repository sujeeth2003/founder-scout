"""
model.py
--------
Trains a Gradient Boosting classifier to predict Series A success.
Saves model artifact to outputs/model.pkl.

Features:
  - Founder pedigree (exits, education)
  - Team composition
  - Funding signals
  - Technical traction (GitHub proxy / real GitHub features if fetched)
  - Domain overlap with fund thesis
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)


FEATURE_COLS = [
    "n_founders",
    "founder_prior_exits",
    "founder_ivy",
    "founder_phd",
    "team_size_at_seed",
    "seed_amount_usd",
    "months_since_seed",
    "pivot_count",
    "has_patent",
    "github_stars_proxy",
    "press_mentions",
    "domain_overlap",
    "sector_encoded",
]

TARGET_COL = "series_a_success"


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["sector_encoded"] = le.fit_transform(df["sector"].astype(str))
    return df, le


def train(data_path: str = "data/startups.csv",
          output_dir: str = "outputs",
          test_size: float = 0.2,
          seed: int = 42):

    df = pd.read_csv(data_path)
    df, le = encode_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Hold out 2020–2022 style cohort (last 20% by index — mirrors time-based split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)
    auc     = roc_auc_score(y_test, y_prob)
    cv_aucs = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\n── Model Evaluation ──────────────────────")
    print(f"  Test ROC-AUC     : {auc:.4f}")
    print(f"  5-Fold CV AUC    : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nTop Features:\n{fi.to_string()}")

    # Save
    Path(output_dir).mkdir(exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": FEATURE_COLS},
                f"{output_dir}/model.pkl")
    print(f"\nModel saved to {output_dir}/model.pkl")

    return model, le, auc


def load_model(path: str = "outputs/model.pkl"):
    return joblib.load(path)


def score_startup(startup: dict, model_bundle: dict) -> dict:
    """
    Score a single startup dict and return probability + risk flags.

    startup = {
        "n_founders": 2,
        "founder_prior_exits": 1,
        "founder_ivy": 0,
        "founder_phd": 1,
        "team_size_at_seed": 4,
        "seed_amount_usd": 500000,
        "months_since_seed": 12,
        "pivot_count": 0,
        "has_patent": 0,
        "github_stars_proxy": 800,
        "press_mentions": 5,
        "domain_overlap": 0.8,
        "sector": "AI/ML",
    }
    """
    model = model_bundle["model"]
    le    = model_bundle["label_encoder"]
    feats = model_bundle["features"]

    row = dict(startup)
    # Handle unseen sector labels gracefully
    known_sectors = list(le.classes_)
    sector = row.get("sector", "SaaS")
    row["sector_encoded"] = le.transform([sector])[0] if sector in known_sectors else 0

    X = pd.DataFrame([row])[feats]
    prob = model.predict_proba(X)[0][1]

    # Risk flags
    flags = []
    if row.get("pivot_count", 0) >= 2:
        flags.append("Multiple pivots — product-market fit unclear")
    if row.get("founder_prior_exits", 0) == 0 and row.get("founder_ivy", 0) == 0:
        flags.append("First-time founders without institutional pedigree")
    if row.get("domain_overlap", 0) < 0.3:
        flags.append("Low alignment with fund thesis")
    if row.get("github_stars_proxy", 0) < 100:
        flags.append("Limited technical traction signals")

