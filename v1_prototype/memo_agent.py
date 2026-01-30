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


@tool
def get_comparable_exits(domain: str) -> str:
    """
    Returns notable exits in this domain as comparables for the memo.
    Replace with Crunchbase API call in production.
    """
    comps = {
        "fintech": "Stripe ($95B), Plaid ($13B acq.), Brex ($12B)",
        "healthtech": "Veeva ($17B IPO), Doximity ($7B IPO), Olive AI ($4B)",
        "ai": "Databricks ($43B), Cohere ($2.2B), Hugging Face ($4.5B)",
        "edtech": "Coursera ($4.8B IPO), Duolingo ($5B IPO)",
    }
    return comps.get(domain.lower(), "No cached comparables. Recommend manual research.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AGENT STATE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MemoState(TypedDict):
    messages: Annotated[list, operator.add]
    startup_profile: dict
    founder_score: dict
    memo: str


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AGENT NODES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SYSTEM_PROMPT = """You are a venture capital analyst. Your job is to write concise, 
structured one-page investment memos for early-stage startups.

Use the tools available to you to pull market size and comparable exits before writing.
Then write the memo in this exact structure:

---
## Investment Memo: [Company Name]

**Domain:** [domain]  
**Stage:** [stage]  
**Founder Score:** [score] ([tier])  

### The Opportunity
[2-3 sentences on the market problem and TAM]

### What They're Building
[2-3 sentences on the product/solution]

### Why This Team
[2-3 sentences on founder signals: prior exits, technical depth, team composition]

### Traction
[1-2 sentences on seed raised, early metrics if available]

### Comparable Exits
[List 2-3 comps with valuations]

### Risk Factors
[2-3 bullet risks]

### Recommendation
[Pass / Watch / Invest â€” one sentence rationale]
---

Be direct. No filler. Write like a GP, not a consultant."""




llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

tools = [search_market_size, get_comparable_exits]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MemoState) -> dict:
    """Main agent: reasons, calls tools, writes memo."""
    profile = state["startup_profile"]
    score = state["founder_score"]

    # Build the user prompt with all context
    context = f"""
Startup Profile:
{json.dumps(profile, indent=2)}

Founder Score: {score['score']} â€” {score['tier']}

Please search for market size and comparable exits for the domain '{profile.get('domain', 'unknown')}', 
then write the investment memo.
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


