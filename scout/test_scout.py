"""Offline checks: python test_scout.py."""

import asyncio
import curses
import json
import sqlite3
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scout import (
    candidates,
    connect,
    create_session,
    defer_paper,
    menu,
    pick,
    session,
    start_session,
)

SINCE = date(2000, 1, 1)


def work(n, published="2020-01-01"):
    return {
        "id": f"https://openalex.org/W{n}",
        "display_name": f"Paper {n}",
        "publication_date": published,
        "publication_year": int(published[:4]) if published else None,
        "cited_by_count": n,
    }


class Client:
    def __init__(self, fail=False):
        self.expanded = []
        self.fail = fail

    async def get(self, path):
        n = int(path.removeprefix("/W"))
        assert n in {1, 2}, "Crawl only seeds and selections"
        self.expanded.append(n)
        return work(n)

    async def references(self, paper):
        if paper["id"].endswith("W1"):
            return [work(2), work(3, "2000-01-01"), work(4, "1999-12-31"), work(5, None)]
        return [work(1), work(3), work(6)]

    async def citers(self, key):
        if key == "W2" and self.fail:
            raise RuntimeError("connection lost")
        return [work(7, "1990-01-01")] if key == "W1" else [work(8)]


class Screen:
    def __init__(self, keys):
        self.keys = iter(keys)
        self.frames = []
        self.frame = []

    def getmaxyx(self):
        return (24, 100)

    def keypad(self, value):
        pass

    def erase(self):
        self.frame = []

    def addnstr(self, row, col, text, *args):
        self.frame.append(text)

    def refresh(self):
        self.frames.append("\n".join(self.frame))

    def get_wch(self):
        return next(self.keys)


def review(selected=(), passed=(), keyword="", stop=False):
    return {"selected": set(selected), "passed": set(passed), "keyword": keyword, "stop": stop}


async def check():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "papers.db"
        client = Client()
        with connect(path) as db:
            first = create_session(db, "Prompt engineering", "W1", SINCE)
            batches = []

            def choose(papers, **kwargs):
                ids = [p["id"] for p in papers]
                batches.append(ids)
                return review(["W2"], ["W3", "W7"]) if len(batches) == 1 else review(passed=ids)

            await session(db, first, choose=choose, client=client)
            assert batches == [["W7", "W3", "W2"], ["W8", "W6"]]
            assert client.expanded == [1, 2]
            second = create_session(db, "Economic agents", "W1", SINCE)
            seen = []

            def stop(papers, **kwargs):
                seen.extend(p["id"] for p in papers)
                return review(stop=True)

            await session(db, second, choose=stop, client=client)
            assert seen == ["W7", "W3", "W2"], (
                "Other sessions' choices and cached expansions must not hide papers"
            )
            assert client.expanded == [1, 2], "Reuse complete neighborhoods across sessions"
            defer_paper(db, second, "W7")
        with connect(path) as db:
            await session(
                db,
                first,
                choose=lambda *a, **k: (_ for _ in ()).throw(AssertionError("Repeated decisions")),
                client=client,
            )
            assert [p["id"] for p in candidates(db, second, SINCE)] == ["W3", "W2"]
            assert db.execute("SELECT paper_id FROM deferred").fetchall() == [("W7",)]
            with patch("builtins.input", side_effect=["d", "1", "Economic modeling", "", ""]):
                deferred_session = start_session(db)
            assert db.execute(
                "SELECT title, seed FROM sessions WHERE id=?", (deferred_session,)
            ).fetchone() == ("Economic modeling", "W7")
            with patch("builtins.input", side_effect=["2"]):
                assert start_session(db) == second, "Startup resumes selected session"

        with connect(Path(tmp) / "interrupted.db") as db:
            sid = create_session(db, "Retry", "W1", SINCE)
            try:
                await session(
                    db,
                    sid,
                    choose=lambda *a, **k: review(["W2"], ["W3", "W7"]),
                    client=Client(fail=True),
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("Expected interrupted crawl")
            resumed = Client()
            await session(
                db,
                sid,
                choose=lambda papers, **k: review(passed=[p["id"] for p in papers]),
                client=resumed,
            )
            assert resumed.expanded == [2]

        with connect(Path(tmp) / "pages.db") as db:
            sid = create_session(db, "All pages", "W999", SINCE)
            with db:
                db.execute(
                    "INSERT INTO papers VALUES ('W999', ?, CURRENT_TIMESTAMP)",
                    (json.dumps(work(999)),),
                )
                for n in range(1, 121):
                    paper = work(n)
                    if n == 1:
                        paper["abstract_inverted_index"] = {"Economic": [0], "modeling": [1]}
                    db.execute(
                        "INSERT INTO papers(id, work) VALUES (?, ?)", (f"W{n}", json.dumps(paper))
                    )
                    db.execute("INSERT INTO citations VALUES ('W999', ?)", (f"W{n}",))

            def defer_and_filter(papers, **options):
                assert len(papers) == 120, "All candidates must reach the picker"
                with patch("curses.curs_set"):
                    return pick(
                        Screen(["n", curses.KEY_LEFT, "/", *"economic", "\n", "q"]),
                        papers,
                        **options,
                    )

            await session(db, sid, choose=defer_and_filter, client=Client())
            assert db.execute("SELECT keyword FROM sessions WHERE id=?", (sid,)).fetchone() == (
                "economic",
            )
            assert db.execute("SELECT paper_id FROM deferred").fetchall() == [("W70",)]
            assert len(candidates(db, sid, SINCE)) == 119, "Deferring must not pass other pages"

        legacy = Path(tmp) / "legacy.db"
        with sqlite3.connect(legacy) as db:
            db.executescript(
                "CREATE TABLE choices (paper_id TEXT PRIMARY KEY, chosen INTEGER); INSERT INTO choices VALUES ('W1',1),('W3',0);"
            )
        for _ in range(2):
            with connect(legacy) as db:
                assert db.execute("SELECT title, seed FROM sessions").fetchall() == [
                    ("Imported exploration", "W1")
                ]
                assert db.execute(
                    "SELECT paper_id, status FROM decisions ORDER BY paper_id"
                ).fetchall() == [("W1", "selected"), ("W3", "passed")]

    papers = [{"id": f"W{n}", "work": work(n)} for n in range(120, 0, -1)]
    papers[-1]["work"]["abstract_inverted_index"] = {"Economic": [0], "modeling": [1]}
    with patch("curses.curs_set"):
        result = pick(Screen([" ", "n", " ", "\n"]), papers)
        assert result["selected"] == {"W120", "W70"}, "Checks survive pagination"
        assert result["passed"] == {f"W{n}" for n in range(21, 70)}, "Only current page is passed"
        screen = Screen(["/", *"ECONOMIC", "\n", " ", "\n"])
        result = pick(screen, papers)
        assert result["selected"] == {"W1"} and not result["passed"]
        assert "1 matches" in screen.frames[-1], "Live filter searches beyond first 50"
        result = pick(
            Screen(["/", *"absent", "\n", "\n", "/", *([curses.KEY_BACKSPACE] * 6), "\n", "q"]),
            papers,
        )
        assert result["stop"] and result["keyword"] == "", "No-match view can clear filter and quit"
        deferred = []
        result = pick(Screen([" ", curses.KEY_LEFT, "q"]), papers, defer=deferred.append)
        assert deferred == ["W120"] and result["stop"], "Left saves immediately even when quitting"
        assert not result["selected"]
        assert pick(Screen([" ", " ", "\n"]), papers)["selected"] == set()
    with patch("builtins.input", side_effect=["n", "21"]):
        assert menu("Items", [(n, str(n)) for n in range(1, 25)]) == 21
    print(
        "PASS: isolated sessions, shared cache, resume/retry, migration, deferred seeds, live filter, pagination, keyboard decisions"
    )


if __name__ == "__main__":
    asyncio.run(check())
