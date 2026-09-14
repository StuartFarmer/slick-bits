"""Offline HTTP transport and nested progress checks."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from rich.console import Console
from rich.progress import Progress

from openalex import OpenAlex, parallel
from scout import connect, scrape_batch


async def check():
    starts = []
    active = peak = attempts = 0

    async def response(request):
        nonlocal active, peak, attempts
        starts.append(time.monotonic())
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.15)
            if request.url.path.endswith("/retry"):
                attempts += 1
                if attempts == 1:
                    return httpx.Response(429, headers={"Retry-After": "0"})
            if request.url.path.endswith("/missing"):
                return httpx.Response(404)
            return httpx.Response(200, json={"id": "https://openalex.org/W1"})
        finally:
            active -= 1

    transport = httpx.MockTransport(response)
    async with OpenAlex() as client:
        original = client.http
        async with httpx.AsyncClient(transport=transport) as http:
            client.http = http
            rows = await parallel(*(client.get(f"/W{n}") for n in range(8)))
            assert len(rows) == 8 and 1 < peak <= 5
            assert all(b - a >= 0.09 for a, b in zip(starts, starts[1:], strict=False)), (
                "Pace all workers together"
            )
            assert await client.get("/retry") == {"id": "https://openalex.org/W1"} and attempts == 2
            try:
                await client.get("/missing")
            except RuntimeError as error:
                assert str(error) == "OpenAlex HTTP 404"
            else:
                raise AssertionError("Non-retryable errors must propagate")
            client.http = original
    assert original.is_closed, "Close pooled connections on exit"

    # Exercise the real progress display and DB writes without making network requests.
    calls = []
    clients = set()

    async def metadata(self, path, params):
        clients.add(id(self))
        calls.append((path, params))
        if path:
            n = int(path.removeprefix("/W"))
            return {"id": f"https://openalex.org/W{n}", "referenced_works": [], "cited_by_count": 1}
        return {
            "results": [{"id": "https://openalex.org/W99"}],
            "meta": {"count": 1, "next_cursor": None},
        }

    progress = Progress("{task.description}", console=Console(record=True), auto_refresh=False)
    with tempfile.TemporaryDirectory() as tmp, connect(Path(tmp) / "test.db") as db:
        with (
            patch("scout.Progress", return_value=progress),
            patch.object(OpenAlex, "_get", metadata),
        ):
            await scrape_batch(["W1", "W2"], db)
        assert len(clients) == 1, "One connection pool for the whole scrape batch"
        assert len(progress.tasks) == 3, "One parent and two reusable child bars"
        parent, refs, citations = progress.tasks
        assert (parent.completed, parent.total) == (2, 2)
        assert (refs.completed, refs.total) == (0, 0)
        assert (citations.completed, citations.total) == (1, 1)
        assert (
            db.execute("SELECT count(*) FROM papers WHERE scraped_at IS NOT NULL").fetchone()[0]
            == 2
        )

    # A failed second paper must preserve the first paper and leave parent progress at 1/2.
    async def broken(self, path, params):
        if path == "/W2":
            raise RuntimeError("network failure")
        return await metadata(self, path, params)

    progress = Progress(auto_refresh=False)
    with tempfile.TemporaryDirectory() as tmp, connect(Path(tmp) / "test.db") as db:
        with patch("scout.Progress", return_value=progress), patch.object(OpenAlex, "_get", broken):
            try:
                await scrape_batch(["W1", "W2"], db)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected failure")
        assert progress.tasks[0].completed == 1
        assert db.execute("SELECT id FROM papers WHERE scraped_at IS NOT NULL").fetchall() == [
            ("W1",)
        ]
    print(
        "PASS: bounded HTTP concurrency, shared pacing, retries/errors, pool cleanup, nested progress and partial-batch persistence"
    )


if __name__ == "__main__":
    asyncio.run(check())
