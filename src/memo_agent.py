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

def enrich_node(state: DealState) -> DealState:
    """Fetch GitHub signals if a username is provided."""
    profile = state["startup_profile"]
    gh_user = profile.get("github_username")

    if gh_user:
        signals = fetch_github_founder(gh_user)
        # Map GitHub features back to model-compatible fields
        profile["github_stars_proxy"] = signals.get("gh_total_stars", 0)
    else:
        signals = {}

    return {**state, "github_signals": signals, "startup_profile": profile}


def score_node(state: DealState) -> DealState:
    """Run the GB classifier and get probability + risk flags."""
    try:
        bundle = load_model("outputs/model.pkl")
        score  = score_startup(state["startup_profile"], bundle)
    except FileNotFoundError:
        # Model not trained yet — return neutral score with note
        score = {
            "series_a_probability": 0.5,
            "score_label": "Unscored (run python src/model.py first)",
            "risk_flags": [],
        }
    return {**state, "model_score": score}


def write_memo_node(state: DealState) -> DealState:
    """Use Claude to write a structured investment memo."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1200)

    profile = state["startup_profile"]
    score   = state["model_score"]
    gh      = state["github_signals"]

    system_prompt = """You are a venture capital analyst. Write concise, precise investment memos.
Output ONLY the memo in clean markdown. No preamble. No meta-commentary.
Structure: Overview | Team | Traction | Market | Risks | Recommendation"""

    user_prompt = f"""
Write a one-page investment memo for this startup.

## Startup Profile
{json.dumps(profile, indent=2)}

## Model Score
- Series A Probability: {score['series_a_probability']:.0%}
- Signal: {score['score_label']}
- Risk Flags: {', '.join(score['risk_flags']) if score['risk_flags'] else 'None identified'}

## GitHub Signals (if available)
{json.dumps(gh, indent=2) if gh else 'Not available'}

Write the memo now. Be direct. Use numbers where available. Flag key risks clearly.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {**state, "memo_markdown": response.content}


# ─── Graph ────────────────────────────────────────────────────────────────────

