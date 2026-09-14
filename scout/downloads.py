"""Download a session's selected papers from their cached public locations."""

import asyncio
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn

from openalex import parallel, work_id


class PDFLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and (attrs.get("name") or "").lower() == "citation_pdf_url":
            self.urls.append(attrs.get("content") or "")
        if tag == "link" and attrs.get("type") == "application/pdf":
            self.urls.append(attrs.get("href") or "")


def locations(work):
    places = [
        work.get("best_oa_location") or {},
        work.get("primary_location") or {},
        *(work.get("locations") or []),
    ]
    urls = [place.get("pdf_url") for place in places]
    urls += [(work.get("open_access") or {}).get("oa_url")]
    urls += [place.get("landing_page_url") for place in places]
    urls += [work.get("doi")]
    return list(
        dict.fromkeys(
            url
            for url in urls
            if isinstance(url, str) and urlsplit(url).scheme in {"http", "https"}
        )
    )


async def retrieve(http, url, target):
    """Save a PDF atomically, or return PDF links advertised by an HTML page."""
    parsed = urlsplit(url)
    if parsed.hostname in {"arxiv.org", "www.arxiv.org"} and parsed.path.startswith("/abs/"):
        url = "https://arxiv.org/pdf/" + parsed.path.removeprefix("/abs/")
    partial = target.with_suffix(".pdf.part")
    try:
        async with http.stream("GET", url) as response:
            response.raise_for_status()
            chunks = response.aiter_bytes(65536)
            first = await anext(chunks, b"")
            if b"%PDF-" in first[:1024]:
                with partial.open("wb") as output:
                    output.write(first)
                    async for chunk in chunks:
                        output.write(chunk)
                partial.replace(target)
                return None
            # Only inspect the head of a landing page, not an unbounded HTML response.
            head = bytearray(first)
            async for chunk in chunks:
                if len(head) >= 1024 * 1024:
                    break
                head.extend(chunk)
            parser = PDFLinks()
            parser.feed(head.decode("utf-8", errors="replace"))
            links = [urljoin(str(response.url), link) for link in parser.urls if link]
            return [link for link in links if urlsplit(link).scheme in {"http", "https"}]
    finally:
        partial.unlink(missing_ok=True)


ANNAS_SCRIPT = Path.home() / "Developer/scraping/annas_pdf.py"


async def retrieve_annas(doi, target):
    """Run the user's resolver as-is; terminate it cleanly on timeout/cancellation."""
    if not ANNAS_SCRIPT.is_file():
        raise RuntimeError(f"Resolver not found: {ANNAS_SCRIPT}")
    partial = target.with_suffix(".annas.part")
    env = dict(os.environ)
    env.pop("ANNAS_DEBUG", None)  # Debug output includes signed mirror URLs.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(ANNAS_SCRIPT),
        doi,
        str(partial),
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            async with asyncio.timeout(90):
                _, stderr = await process.communicate()
        except TimeoutError:
            raise RuntimeError("Anna's resolver exceeded the 90-second time limit") from None
        if process.returncode:
            error = stderr.decode(errors="replace")
            code = re.search(r"\b(403|404|429|500|502|503|504)\b", error)
            reason = f"HTTP {code[1]}" if code else "no PDF returned"
            raise RuntimeError(f"Anna's resolver failed: {reason}")
        if not partial.exists():
            raise RuntimeError("Anna's resolver did not return a PDF")
        with partial.open("rb") as pdf:
            if pdf.read(5) != b"%PDF-":
                raise RuntimeError("Anna's resolver did not return a PDF")
        partial.replace(target)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        partial.unlink(missing_ok=True)


async def download_run(db, session_id, output, *, http=None, annas=False):
    if http is None:
        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=4),
            headers={"User-Agent": "slick-bits/scout"},
        ) as client:
            return await download_run(db, session_id, output, http=client, annas=annas)
    saved = db.execute("SELECT title, seed FROM sessions WHERE id=?", (session_id,)).fetchone()
    if saved is None:
        raise ValueError("Session not found")
    title, seed = saved
    slug = re.sub(r"[^\w-]+", "-", title).strip("-")[:80] or "papers"
    folder = Path(output).expanduser() / f"{session_id}-{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = db.execute(
        """
        SELECT p.id, p.work FROM papers p
        WHERE p.id IN (SELECT paper_id FROM decisions WHERE session_id=? AND status='selected')
           OR p.id=? OR lower(json_extract(p.work, '$.doi'))=?
        ORDER BY p.id
    """,
        (session_id, seed, seed.lower()),
    ).fetchall()
    if not rows:
        raise ValueError("No cached papers in this session yet; explore its seed first")
    manifest = {"session_id": session_id, "title": title, "seed": seed, "papers": []}
    counts = {"downloaded": 0, "existing": 0, "unavailable": 0}
    pacing = asyncio.Lock()

    def save_report():
        partial = folder / "manifest.json.part"
        partial.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        partial.replace(folder / "manifest.json")

    with Progress(
        "{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("PDFs", total=len(rows))

        async def download(key, raw):
            work = json.loads(raw)
            target = folder / f"{work_id(key)}.pdf"
            entry = {
                "id": key,
                "title": work.get("display_name"),
                "doi": work.get("doi"),
                "file": target.name,
            }
            status = "unavailable"
            if target.exists():
                with target.open("rb") as existing:
                    if b"%PDF-" in existing.read(1024):
                        status = "existing"
            if status != "existing" and annas:
                if work.get("doi"):
                    try:
                        await retrieve_annas(work["doi"], target)
                        status = "downloaded"
                        entry["source"] = "annas_pdf.py"
                    except RuntimeError as error:
                        entry["error"] = str(error)
                else:
                    entry["error"] = "No DOI in cached metadata"
            elif status != "existing":
                urls = locations(work)
                original_urls = set(urls)
                seen = set()
                errors = []
                try:
                    async with asyncio.timeout(90):
                        for url in urls:
                            if url in seen:
                                continue
                            seen.add(url)
                            for attempt in range(3):
                                try:
                                    async with pacing:
                                        await asyncio.sleep(0.2)
                                    links = await retrieve(http, url, target)
                                    if links is None:
                                        status = "downloaded"
                                        entry["url"] = url
                                    else:
                                        # Follow one layer of advertised PDF links, not a website crawl.
                                        if url in original_urls:
                                            urls.extend(link for link in links if link not in seen)
                                        errors.append(f"{urlsplit(url).hostname}: no PDF response")
                                    break
                                except httpx.HTTPStatusError as error:
                                    code = error.response.status_code
                                    errors.append(f"{urlsplit(url).hostname}: HTTP {code}")
                                    if code not in {429, 500, 502, 503, 504} or attempt == 2:
                                        break
                                    after = error.response.headers.get("Retry-After", "")
                                    await asyncio.sleep(
                                        float(after) if after.isdigit() else 2**attempt
                                    )
                                except httpx.RequestError:
                                    errors.append(f"{urlsplit(url).hostname}: connection failed")
                                    if attempt < 2:
                                        await asyncio.sleep(2**attempt)
                            if status == "downloaded":
                                break
                except TimeoutError:
                    errors.append("Download exceeded the 90-second limit for this paper")
                if status == "unavailable":
                    entry["error"] = (
                        "; ".join(dict.fromkeys(errors))
                        or "No download location in cached metadata"
                    )
            entry["status"] = status
            counts[status] += 1
            manifest["papers"].append(entry)
            save_report()
            progress.advance(task)

        for start in range(0, len(rows), 4):
            await parallel(*(download(key, raw) for key, raw in rows[start : start + 4]))
    print(
        f"{counts['downloaded']} downloaded; {counts['existing']} already saved; {counts['unavailable']} unavailable.\n{folder}"
    )
    return {**counts, "folder": str(folder)}
