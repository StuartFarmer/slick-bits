"""Apply exact text edits while preserving marked evolution boundaries."""

import re

MARKER = re.compile(r"^.*?EVOLVE-BLOCK-(START|END)[^\r\n]*(?:\r?\n|$)", re.MULTILINE)
DIFF = re.compile(
    r"^<<<<<<< SEARCH\r?\n(.*?)^={5,7}\r?\n(.*?)^>>>>>>> REPLACE(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)
DELIMITER = re.compile(r"^(?:<{6,} SEARCH|>{6,} REPLACE|={5,7})\s*$", re.MULTILINE)


class InvalidCandidate(ValueError):
    """A generated candidate failed an edit or evaluation requirement."""


def evolution_regions(content: str) -> tuple[list[tuple[int, int]], tuple[str, ...]]:
    """Return editable spans and immutable pieces, including marker lines.

    Marker tokens are reserved, language-independent lexical delimiters. A file
    without markers is entirely editable. Nested or unbalanced markers fail.
    """
    regions, skeleton = [], []
    start, previous = None, 0
    for marker in MARKER.finditer(content):
        if marker[1] == "START" and start is None:
            skeleton.append(content[previous : marker.end()])
            start = marker.end()
        elif marker[1] == "END" and start is not None:
            regions.append((start, marker.start()))
            previous, start = marker.start(), None
        else:
            raise InvalidCandidate("Evolution markers must be balanced and non-nested")
    if start is not None:
        raise InvalidCandidate("Unclosed evolution block")
    if not regions:
        return [(0, len(content))], ()
    skeleton.append(content[previous:])
    return regions, tuple(skeleton)


def check_rewrite(parent: str, child: str) -> str:
    """Reject empty, unchanged, or out-of-bounds replacements without normalizing code."""
    if not child.strip() or child == parent:
        raise InvalidCandidate("Candidate is blank or unchanged")
    if evolution_regions(parent)[1] != evolution_regions(child)[1]:
        raise InvalidCandidate("Candidate changed the immutable skeleton or markers")
    return child


def apply_diff(parent: str, response: str) -> str:
    """Apply paper-format SEARCH/REPLACE blocks sequentially and atomically.

    Prose outside blocks is allowed. Searches must be nonempty, unique in the
    current source, and entirely inside one editable span. No fuzzy matching.
    """
    matches = list(DIFF.finditer(response))
    if not matches or DELIMITER.search(DIFF.sub("", response)):
        raise InvalidCandidate("Missing or malformed SEARCH/REPLACE blocks")
    child = parent
    for match in matches:
        # The last newline separates code from the protocol delimiter.
        search, replacement = (re.sub(r"\r?\n$", "", part) for part in match.groups())
        start = child.find(search)
        if not search or start < 0 or child.find(search, start + 1) >= 0:
            raise InvalidCandidate("SEARCH must match exactly one nonempty source segment")
        end = start + len(search)
        if not any(left <= start and end <= right for left, right in evolution_regions(child)[0]):
            raise InvalidCandidate("SEARCH crosses an immutable boundary")
        child = child[:start] + replacement + child[end:]
    return check_rewrite(parent, child)
