"""Offline concurrency, cancellation, paging and HTTP retry checks."""

import asyncio
import time

from openalex import OpenAlex
from scout import fetch


async def check():
    active = peak = 0

    class Client:
        async def get(self, path):
            return {"id": "https://openalex.org/W1"}

        async def references(self, paper):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.04)
                return [{"id": "https://openalex.org/W2"}]
            finally:
                active -= 1

        async def citers(self, key):
            return await self.references({})

    result = await fetch("W1", Client())
    assert peak == 2 and len(result["references"]) == 1, "References and citations must overlap"

    client = OpenAlex()
    active = peak = 0
    calls = []

    async def page(path="", **params):
        nonlocal active, peak
        calls.append(params)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.03)
            number = (
                params["page"]
                if "page" in params
                else (1 if params["cursor"] == "*" else int(params["cursor"]))
            )
            start = (number - 1) * 100
            return {
                "results": [
                    {"id": f"https://openalex.org/W{n}"}
                    for n in range(start + 1, min(start + 100, 601) + 1)
                ],
                "meta": {"count": 601, "next_cursor": str(number + 1) if number < 7 else None},
            }
        finally:
            active -= 1

    client.get = page
    started = time.perf_counter()
    rows = await client.citers("W1")
    elapsed = time.perf_counter() - started
    assert len(rows) == 601 and len({p["id"] for p in rows}) == 601
    assert 1 < peak <= 5, "Numbered pages must overlap in bounded batches"
    assert len(calls) == 7, "Do not spend extra counting requests"
    assert all(not ("cursor" in p and "page" in p) for p in calls)
    print(
        f"Delayed-page benchmark: 7 x 30ms requests in {elapsed:.3f}s; serial minimum 0.210s; peak={peak}"
    )

    async def shifted(path="", **params):
        if params.get("cursor") == "*":
            ids, cursor = range(1, 101), "second"
        elif params.get("cursor") == "second":
            ids, cursor = range(101, 201), "third"
        elif params.get("cursor") == "third" or params.get("page") == 3:
            ids, cursor = [201], None
        else:
            ids, cursor = [100, *range(101, 200)], None
        return {
            "results": [{"id": f"https://openalex.org/W{n}"} for n in ids],
            "meta": {"count": 201, "next_cursor": cursor},
        }

    client.get = shifted
    assert len(await client.citers("W1")) == 201, "Recover gaps if numbered pages shift or overlap"

    events = []
    client.progress = lambda *event: events.append(event)

    async def references(path="", **params):
        ids = params["filter"].split(":", 1)[1].split("|")
        assert len(ids) <= 100
        await asyncio.sleep(0.02 if ids[0] == "W1" else 0.001)
        return {
            "results": [{"id": "https://openalex.org/" + key} for key in ids],
            "meta": {"count": len(ids), "next_cursor": None},
        }

    client.get = references
    rows = await client.references(
        {"referenced_works": [f"https://openalex.org/W{n}" for n in range(1, 540)]}
    )
    assert len(rows) == 539
    completed = [done for label, done, total in events if label == "Reference IDs"]
    assert completed == sorted(completed) and completed[-1] == 539, (
        "Out-of-order reference batches must advance progress monotonically"
    )

    calls.clear()

    async def large(path="", **params):
        assert "page" not in params, "Large result sets need cursor pagination"
        calls.append(params["cursor"])
        return {
            "results": [{"id": f"https://openalex.org/W{len(calls)}"}],
            "meta": {"count": 10001, "next_cursor": "next" if params["cursor"] == "*" else None},
        }

    client.get = large
    assert len(await client.citers("W1")) == 2 and calls == ["*", "next"]

    stopped = asyncio.Event()

    class Broken(Client):
        async def references(self, paper):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def citers(self, key):
            await asyncio.sleep(0)
            raise RuntimeError("failed citation page")

    try:
        await asyncio.wait_for(fetch("W1", Broken()), 0.5)
    except RuntimeError as error:
        assert str(error) == "failed citation page"
    else:
        raise AssertionError("Failed pages must fail the whole neighborhood")
    assert stopped.is_set(), "Cancel and join siblings before closing the client"
    print("PASS: overlapping fetches, bounded pages, cursor fallback, sibling cancellation")


if __name__ == "__main__":
    asyncio.run(check())
