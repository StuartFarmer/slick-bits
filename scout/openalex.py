"""Small OpenAlex client: metadata, references, and citing papers."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import random
import re
import urllib.parse

import httpx
from slick import workflow


def work_id(value: str) -> str:
    identifier = value.rsplit("/", 1)[-1].upper()
    if not re.fullmatch(r"W[0-9]+", identifier):
        raise ValueError(f"Invalid OpenAlex work ID: {value!r}")
    return identifier


def abstract(work: dict) -> str:
    words = {}
    for word, positions in (work.get("abstract_inverted_index") or {}).items():
        for position in positions:
            words[position] = word
    text = html.unescape(" ".join(words[p] for p in sorted(words)))
    text = re.sub(r"<[^>]+>", " ", text).replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", text).strip()


async def parallel(*calls):
    """Join sibling requests before returning or propagating a failure."""
    tasks = [asyncio.ensure_future(call) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class OpenAlex:
    def __init__(self, progress=None, lookup=None):
        self.progress = progress or (lambda label, done, total: None)
        self.lookup = lookup or (lambda key: None)
        self.slots = asyncio.Semaphore(5)
        self.pacing = asyncio.Lock()
        self.next_request = 0.0

    async def __aenter__(self):
        self.http = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=5),
            headers={"User-Agent": "slick-bits/scout"},
        )
        await self.http.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.http.__aexit__(*exc)

    async def _get(self, path: str, params: dict) -> dict:
        credentials = {}
        if os.environ.get("OPENALEX_API_KEY"):
            credentials["api_key"] = os.environ["OPENALEX_API_KEY"]
        if os.environ.get("OPENALEX_MAILTO"):
            credentials["mailto"] = os.environ["OPENALEX_MAILTO"]
        response = await self.http.get(
            "https://api.openalex.org/works" + path, params={**params, **credentials}
        )
        response.raise_for_status()
        return response.json()

    async def get(self, path: str = "", **params) -> dict:
        if path:
            cached = self.lookup(urllib.parse.unquote(path.lstrip("/")))
            if cached is not None:
                return cached
        for attempt in range(8):
            try:
                async with self.slots:
                    # Pace starts across all workers; latency can overlap.
                    async with self.pacing:
                        loop = asyncio.get_running_loop()
                        await asyncio.sleep(max(0, self.next_request - loop.time()))
                        self.next_request = loop.time() + 0.1
                    return await self._get(path, params)
            except httpx.HTTPStatusError as error:
                code = error.response.status_code
                if code not in {429, 500, 502, 503, 504} or attempt == 7:
                    raise RuntimeError(f"OpenAlex HTTP {code}") from None
                after = error.response.headers.get("Retry-After", "")
                delay = float(after) if after.isdigit() else 2**attempt + random.random()
            except (TimeoutError, httpx.RequestError):
                if attempt == 7:
                    raise RuntimeError("OpenAlex request timed out or could not connect") from None
                delay = 2**attempt + random.random()
            logging.getLogger(__name__).info("OpenAlex retry %s/7 in %.1fs", attempt + 1, delay)
            await asyncio.sleep(delay)
        raise RuntimeError("OpenAlex attempts exhausted")

    @workflow
    async def pages(self, params: dict, label: str | None = None) -> list[dict]:
        return await self._pages(params, label)

    async def _pages(self, params, label):
        found = {}

        def collect(page):
            for row in page["results"]:
                found.setdefault(work_id(row["id"]), row)
            if label:
                self.progress(label, len(found), page["meta"].get("count"))

        first = await self.get(**params, per_page=100, cursor="*")
        collect(first)
        total = first["meta"].get("count")
        numbered_pages = (
            isinstance(total, int) and 100 < total <= 10000 and len(first["results"]) == 100
        )
        if numbered_pages:

            async def numbered(number):
                collect(await self.get(**params, per_page=100, page=number))

            for start in range(2, (total + 99) // 100 + 1, 5):
                await parallel(
                    *(numbered(n) for n in range(start, min(start + 5, (total + 99) // 100 + 1)))
                )
        if not numbered_pages or len(found) < total:
            # Cursors cover large sets and recover gaps from shifting numbered pages.
            visited = {"*"}
            page = first
            while page["results"] and (cursor := page["meta"].get("next_cursor")):
                if cursor in visited:
                    raise RuntimeError("OpenAlex repeated a pagination cursor")
                visited.add(cursor)
                page = await self.get(**params, per_page=100, cursor=cursor)
                collect(page)
        if label:
            self.progress(label, len(found), len(found))
        return list(found.values())

    @workflow
    async def references(self, work: dict) -> list[dict]:
        return await self._references(work)

    async def _references(self, work):
        ids = work.get("referenced_works") or []
        self.progress("Reference IDs", 0, len(ids))
        self.progress("Citations", 0, work.get("cited_by_count"))
        found = []
        missing = []
        for identifier in ids:
            cached = self.lookup(work_id(identifier))
            if cached is None:
                missing.append(identifier)
            else:
                found.append(cached)
        cached_count = len(found)
        if cached_count:
            self.progress("Reference IDs", cached_count, len(ids))
        processed = cached_count

        async def batch(chunk):
            nonlocal processed
            page = await self.get(
                filter="openalex_id:" + "|".join(work_id(key) for key in chunk), per_page=100
            )
            found.extend(page["results"])
            processed += len(chunk)
            self.progress("Reference IDs", processed, len(ids))

        for start in range(0, len(missing), 500):
            await parallel(
                *(
                    batch(missing[n : n + 100])
                    for n in range(start, min(start + 500, len(missing)), 100)
                )
            )
        return found

    @workflow
    async def citers(self, identifier: str) -> list[dict]:
        return await self.pages(
            {"filter": f"cites:{work_id(identifier)}", "sort": "cited_by_count:desc"}, "Citations"
        )
