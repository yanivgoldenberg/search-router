#!/usr/bin/env python3
"""MCP server (stdio) wrapping search_router for Claude Code / Codex / any MCP host.

Exposes 8 tools:
  search_serp     - Google SERP cascade
  search_news     - News cascade
  search_ai       - AI-grounded (Tavily/Exa-deep-lite/You.com smart)
  search_deep     - Multi-source synthesis
  search_scrape   - URL → markdown
  search_code     - GitHub + Stack Overflow
  search_academic - OpenAlex/arXiv/PubMed/Crossref
  search_social   - Reddit + HackerNews

Install (claude code):
  claude mcp add search-router -- python3 /config/workspace/scripts/search_router_mcp.py

Install (codex / generic):
  add to mcp config: command="python3", args=["/config/workspace/scripts/search_router_mcp.py"]

Requires: pip install mcp
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from search_router import SearchRouter

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    print("ERROR: install mcp -> pip install mcp", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger("search-router-mcp")

router = SearchRouter()
app: Server = Server("search-router")


def _tool(name: str, description: str, accepts_url: bool = False) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "URL" if accepts_url else "search query"},
                "num": {"type": "integer", "default": 10, "minimum": 1, "maximum": 30},
                "gl": {"type": "string", "default": "us", "description": "geo country code"},
                "hl": {"type": "string", "default": "en", "description": "language code"},
                "no_cache": {"type": "boolean", "default": False},
            },
            "required": ["q"],
        },
    )


TOOLS = {
    "search_serp": ("Google-style organic web SERP. Cascades Serper → SearXNG → Exa → ScrapingBee → SerpAPI → DDG.", False),
    "search_news": ("Fresh news. Cascades Serper(news) → You.com(news) → SearXNG(news) → SerpAPI → HN → Reddit.", False),
    "search_ai": ("AI-grounded answer with citations. Tavily → Exa(deep-lite) → You.com(smart) → DDG-instant → Wikipedia.", False),
    "search_deep": ("Multi-source synthesis. Exa(deep) → Tavily(advanced) → You.com(research) → OpenAlex → arXiv.", False),
    "search_scrape": ("URL → clean markdown. Jina r.jina.ai → self-hosted Firecrawl → ScrapingBee.", True),
    "search_code": ("Code search. GitHub → Stack Overflow → Serper.", False),
    "search_academic": ("Scholarly papers. OpenAlex → arXiv → PubMed → Crossref → Google Scholar.", False),
    "search_social": ("Social discussion. Reddit → HackerNews.", False),
}

TYPE_FROM_TOOL = {f"search_{t}": t for t in ("serp", "news", "ai", "deep", "scrape", "code", "academic", "social")}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [_tool(name, desc, url) for name, (desc, url) in TOOLS.items()]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    type_ = TYPE_FROM_TOOL.get(name)
    if not type_:
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool {name}"}))]
    out = router.run(
        type_,
        arguments["q"],
        num=int(arguments.get("num", 10)),
        gl=arguments.get("gl", "us"),
        hl=arguments.get("hl", "en"),
        bypass_cache=bool(arguments.get("no_cache", False)),
    )
    return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]


async def main() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
