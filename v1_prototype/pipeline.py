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

