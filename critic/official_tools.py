"""Adapt the authors' llm-agent-web-tools Google crawler to an async Slick tool."""

import asyncio
import json

from slick.tools import Tool


def official_google(engine, *, evidence_length: int = 400) -> Tool:
    """Wrap an official Google Search instance; the caller installs/configures it.

    The crawler's synchronous search drives its own event loop, so run it in a
    worker with a dedicated loop. Its own cache, retries, and network policy apply.
    Cancelling the await does not stop the worker's ongoing network request.
    """

    def fetch(query, topk):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = engine.search(query, cache=True, page_cache=True, topk=topk)
            return json.dumps(
                {
                    "title": result.get("title") or "",
                    "url": result.get("link") or "",
                    "page": (result.get("page") or "")[:evidence_length],
                }
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def search(query: str, topk: int = 1) -> str:
        """Search Google for evidence; topk selects the result rank (1 is first)."""
        if topk < 1:
            raise ValueError("The generated result rank must be at least 1")
        return await asyncio.to_thread(fetch, query, topk)

    return Tool(search, name="google", description=search.__doc__)
