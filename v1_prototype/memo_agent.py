"""
Investment Memo Generator
=========================
Takes a scored founder dict + startup profile â†’ generates a structured one-page
investment memo using Groq via LangGraph ReAct agent.

The agent can also call a web_search tool to pull live context on the startup's market.
"""

import os
import json
from typing import TypedDict, Annotated
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TOOLS the agent can call
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@tool
def search_market_size(domain: str) -> str:
    """
    Stub: returns estimated TAM for a given startup domain.
    Replace with Tavily / SerpAPI web search in production.
    """
    mock_markets = {
        "fintech": "Global fintech TAM: ~$310B by 2026 (Mordor Intelligence).",
        "healthtech": "Digital health TAM: ~$660B by 2028 (Grand View Research).",
        "ai": "Enterprise AI TAM: ~$1.8T by 2030 (McKinsey).",
        "edtech": "EdTech TAM: ~$400B by 2026 (HolonIQ).",
    }
    return mock_markets.get(domain.lower(), f"TAM data not cached for '{domain}'. Recommend manual research.")


