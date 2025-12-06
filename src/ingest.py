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

