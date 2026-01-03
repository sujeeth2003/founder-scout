"""
app.py
------
Streamlit demo — paste a startup profile, get a scored investment memo.

Run: streamlit run app.py
"""

import sys
import json
import streamlit as st

sys.path.insert(0, "src")
from src.model import load_model, score_startup, train
from src.memo_agent import generate_memo

