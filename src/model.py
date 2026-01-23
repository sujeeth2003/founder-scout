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


