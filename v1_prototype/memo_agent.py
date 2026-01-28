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

