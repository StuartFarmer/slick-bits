"""Offline checks for the existing Anna's resolver integration."""

import asyncio
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from downloads import download_run, retrieve_annas
from scout import connect, create_session


async def check():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "resolver.py"
        script.write_text(
            "import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_bytes(b'%PDF-1.4\\n%%EOF\\n')\n"
        )
        target = root / "paper.pdf"
        with patch("downloads.ANNAS_SCRIPT", script):
            await retrieve_annas("10.1234/example", target)
            assert target.read_bytes().startswith(b"%PDF-")
            with connect(root / "papers.db") as db:
                sid = create_session(db, "Test", "W1", date(2000, 1, 1))
                with db:
                    db.execute(
                        "INSERT INTO papers(id,work) VALUES ('W1',?)",
                        (
                            json.dumps(
                                {
                                    "id": "https://openalex.org/W1",
                                    "doi": "https://doi.org/10.1234/example",
                                }
                            ),
                        ),
                    )
                result = await download_run(db, sid, root / "files", annas=True)
                assert result["downloaded"] == 1
                result = await download_run(db, sid, root / "files", annas=True)
                assert result["existing"] == 1
        script.write_text(
            "import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text('<html>no PDF</html>')\n"
        )
        original = target.read_bytes()
        with patch("downloads.ANNAS_SCRIPT", script):
            try:
                await retrieve_annas("10.1234/example", target)
            except RuntimeError:
                pass
            else:
                raise AssertionError("HTML must not replace a saved PDF")
        assert target.read_bytes() == original and not list(root.glob("*.part"))
        script.write_text("import time\ntime.sleep(60)\n")
        timeout = asyncio.timeout
        with (
            patch("downloads.ANNAS_SCRIPT", script),
            patch("downloads.asyncio.timeout", lambda seconds: timeout(0.05)),
        ):
            try:
                await retrieve_annas("10.1234/example", target)
            except RuntimeError as error:
                assert "time limit" in str(error)
            else:
                raise AssertionError("Resolver must respect a total deadline")
        assert not list(root.glob("*.part"))
    print(
        "PASS: resolver invocation, run integration/resume, PDF validation, atomic files and subprocess deadline"
    )


if __name__ == "__main__":
    asyncio.run(check())
