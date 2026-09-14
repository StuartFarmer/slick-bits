"""Give OpenAlex a paper ID; save the paper, its references, and its citations."""

import argparse
import asyncio
import curses
import json
import re
import sqlite3
import textwrap
from contextlib import closing, contextmanager
from datetime import date
from pathlib import Path
from urllib.parse import quote

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from slick import workflow

from downloads import download_run
from openalex import OpenAlex, abstract, parallel, work_id


def identifier(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"(?:https?://openalex\.org/)?W\d+", value, re.I):
        return work_id(value)
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I)
    if re.fullmatch(r"10\.\d{4,9}/\S+", doi) and not any(c in doi for c in "?#"):
        return "https://doi.org/" + doi.lower()
    raise ValueError("Enter a DOI or OpenAlex W… ID")


@contextmanager
def connect(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, work TEXT NOT NULL, scraped_at TEXT);
            CREATE TABLE IF NOT EXISTS citations (citing TEXT, cited TEXT, PRIMARY KEY(citing, cited));
            CREATE INDEX IF NOT EXISTS citations_cited ON citations(cited);
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY, title TEXT NOT NULL, seed TEXT NOT NULL,
                since TEXT NOT NULL, keyword TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS decisions (
                session_id INTEGER NOT NULL REFERENCES sessions(id), paper_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('selected', 'passed', 'deferred')),
                PRIMARY KEY(session_id, paper_id)
            );
            CREATE TABLE IF NOT EXISTS deferred (
                paper_id TEXT PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES sessions(id),
                saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)
        db.execute("PRAGMA foreign_keys = ON")
        if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'choices'").fetchone():
            seed = db.execute(
                "SELECT paper_id FROM choices WHERE chosen = 1 ORDER BY rowid LIMIT 1"
            ).fetchone()
            if seed and not db.execute("SELECT 1 FROM sessions").fetchone():
                with db:
                    sid = db.execute(
                        "INSERT INTO sessions(title, seed, since) VALUES ('Imported exploration', ?, '2000-01-01')",
                        seed,
                    ).lastrowid
                    db.execute(
                        "INSERT INTO decisions SELECT ?, paper_id, CASE chosen WHEN 1 THEN 'selected' ELSE 'passed' END FROM choices",
                        (sid,),
                    )
        # Keep papers already collected by the previous app. Its other tables stay untouched.
        if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'scout_papers'").fetchone():
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO papers(id, work) SELECT id, work FROM scout_papers"
                )
        yield db


@workflow
async def fetch(value: str, client) -> dict:
    paper = await client.get("/" + quote(identifier(value), safe="/:"))
    references, citations = await parallel(
        client.references(paper), client.citers(work_id(paper["id"]))
    )
    return {"paper": paper, "references": references, "citations": citations}


async def scrape_batch(values, db, client=None) -> list[dict]:
    if client is not None:
        return [await scrape(value, db, client) for value in values]
    if not values:
        return []

    def lookup(key):
        row = db.execute("SELECT work FROM papers WHERE id = ?", (key,)).fetchone()
        if row is None and key.startswith("https://doi.org/"):
            row = db.execute(
                "SELECT work FROM papers WHERE lower(json_extract(work, '$.doi')) = ?",
                (key.lower(),),
            ).fetchone()
        return json.loads(row[0]) if row else None

    with Progress(
        "{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()
    ) as progress:
        overall = progress.add_task("Papers", total=len(values))
        tasks = {
            label: progress.add_task(f"  {label}", total=None)
            for label in ("Reference IDs", "Citations")
        }

        def report(label, done, total):
            progress.update(tasks[label], completed=done, total=total, visible=True, refresh=True)

        results = []
        async with OpenAlex(progress=report, lookup=lookup) as api:
            for value in values:
                progress.update(overall, description=f"Papers · {value}")
                for task in tasks.values():
                    progress.reset(task, total=0, visible=False)
                results.append(await scrape(value, db, api))
                progress.advance(overall)
        return results


async def scrape(value: str, db, client=None) -> dict:
    if client is None:
        return (await scrape_batch([value], db))[0]
    result = await fetch(value, client)
    root = work_id(result["paper"]["id"])
    # ponytail: one neighborhood in memory; stream pages into SQLite if this outgrows RAM.
    works = [result["paper"], *result["references"], *result["citations"]]
    with db:
        db.executemany(
            """
            INSERT INTO papers(id, work) VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET work = excluded.work
        """,
            [(work_id(work["id"]), json.dumps(work, ensure_ascii=False)) for work in works],
        )
        links = [(root, work_id(work["id"])) for work in result["references"]]
        links += [(work_id(work["id"]), root) for work in result["citations"]]
        db.executemany("INSERT OR IGNORE INTO citations VALUES (?, ?)", links)
        db.execute("UPDATE papers SET scraped_at = CURRENT_TIMESTAMP WHERE id = ?", (root,))
    return {
        "id": root,
        "references": len(result["references"]),
        "citations": len(result["citations"]),
    }


def create_session(db, title: str, seed: str, since: date, keyword: str = "") -> int:
    title = title.strip()
    if not title:
        raise ValueError("Give the session a title describing your search")
    seed = identifier(seed)
    with db:
        return db.execute(
            "INSERT INTO sessions(title, seed, since, keyword) VALUES (?, ?, ?, ?)",
            (title, seed, since.isoformat(), keyword.strip()),
        ).lastrowid


def candidates(db, session_id: int, since: date) -> list[dict]:
    found = {}
    for (root,) in db.execute(
        "SELECT paper_id FROM decisions WHERE session_id = ? AND status = 'selected'",
        (session_id,),
    ):
        rows = db.execute(
            """
            SELECT p.id, p.work, c.citing FROM citations c
            JOIN papers p ON p.id = CASE WHEN c.citing = ? THEN c.cited ELSE c.citing END
            WHERE (c.citing = ? OR c.cited = ?)
                AND NOT EXISTS (SELECT 1 FROM decisions WHERE session_id = ? AND paper_id = p.id)
            """,
            (root, root, root, session_id),
        )
        for key, raw, citing in rows:
            paper = json.loads(raw)
            if citing == root:  # The cutoff applies to references, not forward citations.
                try:
                    published = date.fromisoformat(paper.get("publication_date") or "")
                except ValueError:
                    continue
                if published < since:
                    continue
            found[key] = {"id": key, "work": paper}
    return sorted(found.values(), key=lambda p: (-(p["work"].get("cited_by_count") or 0), p["id"]))


def defer_paper(db, session_id: int, paper_id: str) -> None:
    with db:
        db.execute(
            "INSERT OR IGNORE INTO deferred(paper_id, session_id) VALUES (?, ?)",
            (paper_id, session_id),
        )
        db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, 'deferred')", (session_id, paper_id)
        )


def pick(screen, papers: list[dict], *, keyword="", title="Explore papers", defer=None) -> dict:
    """Review 50 per page. Hidden pages stay undecided; deferrals save immediately."""
    curses.curs_set(0)
    screen.keypad(True)

    def line(row, text, style=0):
        height, width = screen.getmaxyx()
        if row < height - 1:
            try:
                screen.addnstr(row, 0, text, max(1, width - 1), style)
            except curses.error:
                pass  # A resize or wide character can reach the terminal's edge.

    # ponytail: keep candidate text in memory; use SQLite FTS if lists outgrow RAM.
    searchable = {
        p["id"]: ((p["work"].get("display_name") or "").casefold(), abstract(p["work"]).casefold())
        for p in papers
    }
    cursor, page, selected, deferred, editing = 0, 0, set(), set(), False
    while True:
        needle = keyword.strip().casefold()
        matches = [
            p
            for p in papers
            if p["id"] not in deferred and any(needle in text for text in searchable[p["id"]])
        ]
        pages = max(1, (len(matches) + 49) // 50)
        page = min(page, pages - 1)
        current_page = matches[page * 50 : (page + 1) * 50]
        cursor = min(cursor, max(0, len(current_page) - 1))
        height, width = screen.getmaxyx()
        visible = max(1, height - 11)
        top = cursor // visible * visible
        screen.erase()
        line(0, title)
        line(
            1,
            f"{len(matches)} matches · page {page + 1}/{pages} · {len(selected)} checked · most cited first",
        )
        line(2, "Up/Down: move | Space: check | Left: defer (saved) | n/p or PgDn/PgUp: page")
        line(3, "/: filter | Enter: crawl checked + pass this page's rest | q: menu")
        line(
            4,
            f"Filter: {keyword}{'_' if editing else ''}"
            + ("  [type to filter; Enter/Esc: done]" if editing else ""),
        )
        for index in range(top, min(top + visible, len(current_page))):
            item = current_page[index]
            paper = item["work"]
            label = f"[{'x' if item['id'] in selected else ' '}] {paper.get('cited_by_count') or 0:>6} cites | {paper.get('publication_year') or '?'} | {paper.get('display_name') or 'Untitled'}"
            line(index - top + 5, label, curses.A_REVERSE if index == cursor else 0)
        if current_page:
            current = current_page[cursor]
            line(visible + 6, f"{current['id']} · {current['work'].get('doi') or ''}")
            summary = abstract(current["work"]) or "No abstract available."
            for offset, text in enumerate(textwrap.wrap(summary, max(1, width - 2))[:3]):
                line(visible + 7 + offset, text)
        else:
            line(5, "No matches. Press / to change or clear the filter, or q for the menu.")
        screen.refresh()
        key = screen.get_wch()
        if editing:
            if key in ("\n", "\r", "\x1b", curses.KEY_ENTER):
                editing = False
            elif key in ("\x7f", "\b", curses.KEY_BACKSPACE):
                keyword = keyword[:-1]
            elif key == "\x15":  # Ctrl-U clears the filter.
                keyword = ""
            elif isinstance(key, str) and key.isprintable():
                keyword += key
            cursor, page = 0, 0
            continue
        if key == "/":
            editing = True
        elif key in ("q", "\x1b"):
            return {"selected": set(), "passed": set(), "keyword": keyword, "stop": True}
        elif key in (curses.KEY_NPAGE, "n"):
            page, cursor = min(page + 1, pages - 1), 0
        elif key in (curses.KEY_PPAGE, "p"):
            page, cursor = max(page - 1, 0), 0
        elif key in (curses.KEY_DOWN, "j"):
            cursor = min(cursor + 1, max(0, len(current_page) - 1))
        elif key in (curses.KEY_UP, "k"):
            cursor = max(cursor - 1, 0)
        elif key == " " and current_page:
            selected.symmetric_difference_update({current_page[cursor]["id"]})
        elif key == curses.KEY_LEFT and current_page and defer:
            paper_id = current_page[cursor]["id"]
            defer(paper_id)
            deferred.add(paper_id)
            selected.discard(paper_id)
        elif key in ("\n", "\r", curses.KEY_ENTER) and (current_page or selected):
            return {
                "selected": selected,
                "passed": {p["id"] for p in current_page} - selected,
                "keyword": keyword,
                "stop": False,
            }


async def session(db, session_id: int, *, choose=None, client=None) -> None:
    saved_session = db.execute(
        "SELECT title, seed, since, keyword FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if saved_session is None:
        raise ValueError("Session not found")
    title, seed, since, keyword = saved_session
    since = date.fromisoformat(since)
    saved = db.execute(
        "SELECT id FROM papers WHERE (id = ? OR lower(json_extract(work, '$.doi')) = ?) AND scraped_at IS NOT NULL",
        (seed, seed.lower()),
    ).fetchone()
    if saved:
        root = saved[0]
    else:
        root = (await scrape(seed, db, client))["id"]
    with db:
        db.execute("UPDATE sessions SET seed = ? WHERE id = ?", (root, session_id))
        db.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, 'selected')", (session_id, root))
    while True:
        pending = db.execute(
            """SELECT p.id FROM papers p JOIN decisions d ON d.paper_id = p.id
            WHERE d.session_id = ? AND d.status = 'selected' AND p.scraped_at IS NULL ORDER BY p.rowid""",
            (session_id,),
        ).fetchall()
        await scrape_batch([key for (key,) in pending], db, client)
        papers = candidates(db, session_id, since)
        if not papers:
            print("No new papers to review. Session saved.")
            return

        def save_deferred(key):
            defer_paper(db, session_id, key)

        options = {"keyword": keyword, "title": title, "defer": save_deferred}
        result = choose(papers, **options) if choose else curses.wrapper(pick, papers, **options)
        keyword = result["keyword"].strip()
        with db:
            db.execute("UPDATE sessions SET keyword = ? WHERE id = ?", (keyword, session_id))
            if not result["stop"]:
                db.executemany(
                    "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
                    [
                        (session_id, key, status)
                        for status, keys in (
                            ("selected", result["selected"]),
                            ("passed", result["passed"]),
                        )
                        for key in keys
                    ],
                )
        if result["stop"]:
            print("Session saved. Unconfirmed checks discarded; deferred papers kept.")
            return
        print(f"Selected {len(result['selected'])}; passed {len(result['passed'])}.", flush=True)


def cutoff(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a date such as 2000-01-01") from error


def menu(title, items, extra=""):
    """Numbered, paged menu shared by sessions and deferred seeds."""
    page = 0
    while True:
        print(f"\n{title} · page {page + 1}/{max(1, (len(items) + 19) // 20)}")
        for i in range(page * 20, min((page + 1) * 20, len(items))):
            print(f"  {i + 1}. {items[i][1]}")
        if not items:
            print("  Nothing saved yet.")
        choice = input(f"Number | n/p: page | {extra}q: back/quit: ").strip().lower()
        if choice == "q":
            return None
        if choice in ("new", "d", "download") and extra:
            return choice
        if choice == "n":
            page = min(page + 1, max(0, (len(items) - 1) // 20))
        elif choice == "p":
            page = max(0, page - 1)
        elif choice.isdecimal() and page * 20 < int(choice) <= min((page + 1) * 20, len(items)):
            return items[int(choice) - 1][0]
        else:
            print("Choose a number on this page or one of the listed commands.")


def start_session(db, *, seed=None, title=None, since=None, keyword=None, download_dir=None):
    if seed is None:
        while True:
            items = [
                (sid, f"{name} [{paper}]")
                for sid, name, paper in db.execute(
                    "SELECT id, title, seed FROM sessions ORDER BY id DESC"
                )
            ]
            choice = menu(
                "Search sessions", items, "new: new search | d: deferred papers | download: PDFs | "
            )
            if choice is None or isinstance(choice, int):
                return choice
            if choice == "new":
                break
            if choice == "download":
                selected_run = menu("Download seed and selected papers from a run", items)
                if selected_run is not None:
                    asyncio.run(
                        download_run(
                            db, selected_run, download_dir or Path(__file__).parent / "downloads"
                        )
                    )
                continue
            deferred = [
                (key, f"{json.loads(raw).get('display_name') or key} [{key}] — from {origin}")
                for key, raw, origin in db.execute(
                    "SELECT d.paper_id, p.work, s.title FROM deferred d JOIN papers p ON p.id = d.paper_id JOIN sessions s ON s.id = d.session_id ORDER BY d.saved_at DESC, d.paper_id"
                )
            ]
            seed = menu("Deferred papers — choose a seed for a new search", deferred)
            if seed:
                break
    while seed is None:
        try:
            seed = identifier(input("OpenAlex ID or DOI: "))
        except ValueError as error:
            print(error)
    while not title or not title.strip():
        title = input("Search title / goal: ").strip()
    while since is None:
        try:
            since = cutoff(
                input("References published on/after [2000-01-01]: ").strip() or "2000-01-01"
            )
        except argparse.ArgumentTypeError as error:
            print(error)
    if keyword is None:
        keyword = input("Initial title/abstract filter [all; change with /]: ").strip()
    return create_session(db, title, seed, since, keyword)


def main():
    parser = argparse.ArgumentParser(
        description="Named OpenAlex searches: select, defer, and explore papers."
    )
    parser.add_argument(
        "paper", nargs="?", help="seed for a new session (omit to choose a saved session)"
    )
    parser.add_argument("--title", help="title / goal for a new session")
    parser.add_argument(
        "--since", type=cutoff, help="reference cutoff for a new session (default: 2000-01-01)"
    )
    parser.add_argument("--keyword", help="initial title/abstract filter for a new session")
    parser.add_argument("--db", type=Path, default=Path(__file__).parent / "data" / "scout.db")
    parser.add_argument(
        "--download",
        type=int,
        metavar="SESSION_ID",
        help="download a run's seed and selected papers, then exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "downloads",
        help="root folder for PDF downloads",
    )
    parser.add_argument(
        "--annas",
        action="store_true",
        help="with --download: use ~/Developer/scraping/annas_pdf.py for missing PDFs",
    )
    args = parser.parse_args()
    if args.annas and args.download is None:
        parser.error("--annas requires --download SESSION_ID")
    try:
        with connect(args.db) as db:
            if args.download is not None:
                asyncio.run(download_run(db, args.download, args.output, annas=args.annas))
                return
            sid = start_session(
                db,
                seed=args.paper,
                title=args.title,
                since=args.since,
                keyword=args.keyword,
                download_dir=args.output,
            )
            while sid is not None:
                asyncio.run(session(db, sid))
                sid = start_session(db, download_dir=args.output)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Confirmed decisions, deferred papers, and completed fetches are saved.")
    except (ValueError, RuntimeError, OSError, curses.error) as error:
        parser.exit(1, f"{error}\nProgress saved. Restart and choose your session to continue.\n")


if __name__ == "__main__":
    main()
