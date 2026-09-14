"""Offline progress check: python test_progress.py."""

import asyncio

from openalex import OpenAlex


async def check():
    events = []
    client = OpenAlex(progress=lambda *event: events.append(event))

    async def page(path="", **params) -> dict:
        if params["filter"].startswith("openalex_id:"):
            ids = params["filter"].split(":", 1)[1].split("|")
            return {
                "results": [{"id": "https://openalex.org/" + key} for key in ids],
                "meta": {"count": len(ids), "next_cursor": None},
            }
        if params["cursor"] == "*":
            return {
                "results": [{"id": "https://openalex.org/W101"}],
                "meta": {"count": 2, "next_cursor": "next"},
            }
        return {
            "results": [{"id": "https://openalex.org/W101"}, {"id": "https://openalex.org/W102"}],
            "meta": {"count": 2, "next_cursor": None},
        }

    client.get = page
    await client.references(
        {
            "referenced_works": [f"https://openalex.org/W{n}" for n in range(1, 52)],
            "cited_by_count": 10,
        }
    )
    assert events == [
        ("Reference IDs", 0, 51),
        ("Citations", 0, 10),
        ("Reference IDs", 51, 51),
    ]
    events.clear()
    result = await client.citers("W1")
    assert len(result) == 2
    assert events[0] == ("Citations", 1, 2), "First page corrects the metadata estimate"
    assert events[-1] == ("Citations", 2, 2)
    assert all(done <= 2 for _, done, _ in events), "Duplicate rows must not inflate progress"
    events.clear()
    await client.references({})
    assert events == [("Reference IDs", 0, 0), ("Citations", 0, None)]
    cached = {"id": "https://openalex.org/W1", "display_name": "Cached paper"}
    client = OpenAlex(
        progress=lambda *event: events.append(event),
        lookup=lambda key: cached if key == "W1" else None,
    )
    assert await client.get("/W1") == cached

    async def request(path, params):
        assert params["filter"] == "openalex_id:W2", "Cached references must not be requested again"
        return {
            "results": [{"id": "https://openalex.org/W2"}],
            "meta": {"count": 1, "next_cursor": None},
        }

    client._get = request
    events.clear()
    rows = await client.references(
        {"referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"]}
    )
    assert len(rows) == 2 and events[-1] == ("Reference IDs", 2, 2)
    assert ("Reference IDs", 1, 2) in events
    print("PASS: initial totals, reference batches, citation pages, duplicates, empty lists")


if __name__ == "__main__":
    asyncio.run(check())
