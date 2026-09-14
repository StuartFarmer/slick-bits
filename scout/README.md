# OpenAlex explorer

Named, interactive research sessions over a shared OpenAlex paper cache.

```bash
cd ~/Developer/AI/slick-bits/scout
.venv/bin/python scout.py
# Or start a new named search directly:
.venv/bin/python scout.py W4393160302 --title "Prompt engineering techniques" --since 2000-01-01 --keyword "prompt"
```

At startup, choose a saved session, enter `new` to start one, or `d` to browse
deferred papers. Each session has a title describing its goal, a seed DOI or
OpenAlex ID, a reference cutoff (default **2000-01-01**), and a saved keyword filter.
Selecting a deferred paper starts a new session and asks for its title.
The numbered menus use `n`/`p` to page and `q` to go back or quit.

While reviewing papers:

- **Up/Down** (or `j`/`k`): move through the current page.
- **Space**: check/uncheck a paper to crawl next.
- **n/p** or **Page Down/Up**: browse pages of 50, most cited first. Short terminals
  scroll within each page as you move. Checks survive page and filter changes.
- **/**: edit the title/abstract filter. Results update as you type across all
  cached candidates, before pagination. Matching is case-insensitive literal text.
  **Ctrl-U** clears it; **Enter/Esc** finishes editing. A blank filter shows all.
- **Left**: immediately save the highlighted paper to the shared deferred list and
  remove it from this session's candidates. It won't be crawled in this session.
  The deferred list records the title of the session where you first saved it.
- **Enter**: save all checked papers (including other pages or hidden matches),
  pass only the current page's unchecked papers, and crawl the selections.
  Other pages and filtered-out papers stay undecided. With nothing checked,
  this passes the current page and continues reviewing remaining candidates.
- **q/Esc**: return to the session menu. Unconfirmed checks are discarded;
  deferrals, confirmed decisions, and the current filter are saved.

Selections, passes, and deferrals are scoped to each session. A paper passed in a
prompt-engineering search can still be considered in an economics search. Cached
metadata and completed neighborhoods are shared, so reusing a seed or selecting
an already-crawled paper doesn't fetch its neighborhood again. Interrupted crawls
resume from the last completed neighborhood when you reopen the session.

The inclusive publication-date cutoff applies only to **references**; references
without a publication date are omitted from review. Forward citations are
unaffected. All fetched metadata, including older references, remains cached.
Citation counts are OpenAlex's `cited_by_count` at retrieval time.

All API pages are fetched and cached. A parent progress bar counts completed
papers in the current scrape batch; the two indented bars show reference IDs
processed and citing papers fetched for the current paper. Child bars reset as
each paper starts. The parent advances only after its neighborhood is saved.

Requests use a shared HTTPX async connection pool for each scrape batch, capped
at five in flight and paced to ten starts per second. References and citations
fetch concurrently, reference lookups batch up to 100 IDs, and numbered citation
pages fetch concurrently for result sets up to 10,000. Larger sets retain cursor
paging so every page can still be retrieved. Retries/backoff and cache reuse
remain enabled. Failed requests cancel sibling tasks before the pool closes.
Citation totals update from the API's page metadata.
No model is used. Set `OPENALEX_API_KEY` for API authentication and
`OPENALEX_MAILTO` to identify your requests.

`--db PATH` chooses a database (default: `data/scout.db`). Older global choices are
imported once into **Imported exploration**; their original tables remain intact.
Because the old database had no session titles or boundaries, they become one
session using its first selected paper as the seed and the default date cutoff.
Unreviewed cached papers are now available through pagination.

Install: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt`.
The picker uses Python's standard `curses` module on macOS/Linux.

Offline checks:

```bash
.venv/bin/python test_scout.py
.venv/bin/python test_progress.py
.venv/bin/python test_concurrency.py
.venv/bin/python test_http.py
```

The concurrency check includes a delayed-response benchmark; it measures scheduling
overlap, not live OpenAlex latency. API constraints: [paging](https://help.openalex.org/api/paging/)
and [request limits / batching](https://help.openalex.org/api/authentication/).

## Download a run

Enter `download` in the startup menu, then choose a named run. Or download directly:

```bash
.venv/bin/python scout.py --download 2
# Optional destination:
.venv/bin/python scout.py --download 2 --output ~/Downloads/papers
```

Downloads include the seed and all papers selected in that session. PDFs go into
`downloads/<session-id>-<title>/`, named by OpenAlex ID. `manifest.json` maps files
to titles and DOIs and records unavailable papers and their errors. Re-running
skips existing PDFs and retries missing ones. Each file is written atomically;
completed PDFs survive interruption. Each paper has a 90-second total download
time limit, so a stalled host cannot hold up the rest of the run.

The downloader tries cached PDF locations and PDF links advertised by landing
pages. Four downloads can run concurrently through one HTTP client, with paced
requests and retries. HTML responses are not saved as PDFs. Availability depends
on the public locations; inaccessible papers remain listed in the manifest.

Offline check: `.venv/bin/python test_downloads.py`.

To retry missing PDFs using your DOI resolver (including its browser headers and
referrer handling):

```bash
.venv/bin/python scout.py --download 2 --annas
```

This runs `~/Developer/scraping/annas_pdf.py` directly for each missing PDF,
skips files already saved, and updates the same manifest. Each resolver gets
90 seconds; cancelled processes are stopped before cleaning up temporary files.
The resolver's existing optional cookie support is preserved. No cookie is
required by Scout to attempt the download.

Offline resolver integration check: `.venv/bin/python test_annas.py`.
