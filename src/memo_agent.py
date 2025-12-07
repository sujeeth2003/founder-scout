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

