"""Offline PDF download checks."""

import asyncio
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx

from downloads import download_run
from scout import connect, create_session

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


async def check():
    calls = []

    async def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/landing":
            return httpx.Response(200, text='<meta name="citation_pdf_url" content="/paper.pdf">')
        if request.url.path == "/denied":
            return httpx.Response(403)
        if request.url.path == "/html.pdf":
            return httpx.Response(200, text="<html>Not a PDF</html>")
        return httpx.Response(200, content=PDF)

    with tempfile.TemporaryDirectory() as tmp, connect(Path(tmp) / "papers.db") as db:
        sid = create_session(db, "Prompt / techniques", "W1", date(2000, 1, 1))
        other = create_session(db, "Other", "W9", date(2000, 1, 1))
        with db:
            for n, locs in [
                (
                    1,
                    [
                        {"pdf_url": "https://test/denied"},
                        {"landing_page_url": "https://test/landing"},
                    ],
                ),
                (2, [{"pdf_url": "https://test/html.pdf"}]),
                (3, []),
                (4, [{"pdf_url": "https://test/passed.pdf"}]),
                (9, [{"pdf_url": "https://test/other.pdf"}]),
            ]:
                db.execute(
                    "INSERT INTO papers(id,work) VALUES (?,?)",
                    (
                        f"W{n}",
                        json.dumps(
                            {
                                "id": f"https://openalex.org/W{n}",
                                "display_name": f"Paper {n}",
                                "locations": locs,
                            }
                        ),
                    ),
                )
            db.executemany(
                "INSERT INTO decisions VALUES (?,?,?)",
                [
                    (sid, "W1", "selected"),
                    (sid, "W2", "selected"),
                    (sid, "W3", "selected"),
                    (sid, "W4", "passed"),
                    (other, "W9", "selected"),
                ],
            )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        ) as http:
            folder = Path(tmp) / "downloads"
            result = await download_run(db, sid, folder, http=http)
            assert result["downloaded"] == 1 and result["unavailable"] == 2
            saved = list(folder.rglob("*.pdf"))
            assert len(saved) == 1 and saved[0].read_bytes() == PDF
            assert "/passed.pdf" not in calls and "/other.pdf" not in calls
            report = json.loads(next(folder.rglob("manifest.json")).read_text())
            assert len(report["papers"]) == 3 and report["title"] == "Prompt / techniques"
            assert not list(folder.rglob("*.part"))
            calls.clear()
            result = await download_run(db, sid, folder, http=http)
            assert result["existing"] == 1 and "/paper.pdf" not in calls, "Resume skips saved PDFs"
    timeout = asyncio.timeout

    async def stall(*args):
        await asyncio.Event().wait()

    with tempfile.TemporaryDirectory() as tmp, connect(Path(tmp) / "bounded.db") as db:
        sid = create_session(db, "Slow host", "W1", date(2000, 1, 1))
        with db:
            db.execute(
                "INSERT INTO papers(id,work) VALUES ('W1',?)",
                (
                    json.dumps(
                        {
                            "id": "https://openalex.org/W1",
                            "locations": [{"pdf_url": "https://test/slow"}],
                        }
                    ),
                ),
            )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            with (
                patch("downloads.retrieve", stall),
                patch("downloads.asyncio.timeout", lambda seconds: timeout(0.25)),
            ):
                result = await download_run(db, sid, Path(tmp) / "files", http=http)
            assert result["unavailable"] == 1, "A stalled paper must not block the run forever"
            report = json.loads(next((Path(tmp) / "files").rglob("manifest.json")).read_text())
            assert "90-second limit" in report["papers"][0]["error"]
    print(
        "PASS: selected run only, alternate links, landing-page PDF links, HTML rejection, missing PDFs, manifest and resume"
    )


if __name__ == "__main__":
    asyncio.run(check())
