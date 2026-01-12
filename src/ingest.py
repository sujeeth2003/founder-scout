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

