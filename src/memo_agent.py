"""
memo_agent.py
-------------
LangGraph agent that takes a startup profile + model score and
auto-generates a structured one-page investment memo.

Flow:
  [Input] startup profile
      ↓
  [Node: enrich]     → fetch GitHub signals if github_username present
      ↓
  [Node: score]      → run GB model, get probability + risk flags
      ↓
  [Node: research]   → web search for press coverage / competitors (optional)
      ↓
  [Node: write_memo] → LLM generates structured investment memo
      ↓
  [Output] memo dict + markdown string

Requires: ANTHROPIC_API_KEY in environment
Optional:  GITHUB_TOKEN for higher rate limits
"""

import os
import json
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ingest import fetch_github_founder
from model import load_model, score_startup
from dotenv import load_dotenv
load_dotenv()

# ─── State ────────────────────────────────────────────────────────────────────

class DealState(TypedDict):
    startup_profile:  dict          # raw input
    github_signals:   dict          # from GitHub API
    model_score:      dict          # probability + flags
    memo_markdown:    str           # final output
    error:            Optional[str]


# ─── Nodes ────────────────────────────────────────────────────────────────────

