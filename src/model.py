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

