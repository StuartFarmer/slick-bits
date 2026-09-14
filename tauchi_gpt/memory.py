"""Chunk caller-selected text and retrieve source or result passages by cosine similarity."""

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Embed = Callable[[Sequence[str]], Awaitable[Sequence[Sequence[float]]]]


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    start: int
    end: int
    digest: str
    kind: Literal["document", "result"]


class LocalMemory:
    """An in-memory vector store; the caller owns embedding locality and document loading."""

    def __init__(self, embed: Embed, *, chunk_size: int = 2000, overlap: int = 200):
        self.embed = embed
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.vectors: list[np.ndarray] = []

    async def add(self, document: Document, prefix: str, kind: Literal["document", "result"]):
        """Commit chunks only after their entire embedding batch passes validation."""
        chunks = []
        digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        for start in range(0, len(document.text), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(document.text))
            chunks.append(
                Chunk(
                    f"{prefix}:{start}",
                    document.source,
                    document.text[start:end],
                    start,
                    end,
                    digest,
                    kind,
                )
            )
            if end == len(document.text):
                break
        if chunks:
            vectors = await self._encode([chunk.text for chunk in chunks])
            self.chunks.extend(chunks)
            self.vectors.extend(vectors)

    async def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(await self.embed(texts), dtype=float)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(texts)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("embedding model returned an invalid matrix")
        norms = np.linalg.norm(vectors, axis=1)
        if (norms == 0).any() or not np.isfinite(norms).all():
            raise ValueError("embedding model returned zero or overflowing vectors")
        if self.vectors and vectors.shape[1] != len(self.vectors[0]):
            raise ValueError("embedding dimensions changed")
        return vectors / norms[:, None]

    async def retrieve(self, query: str, top_k: int = 5) -> tuple[Chunk, ...]:
        if not self.chunks or top_k == 0:
            return ()
        vector = (await self._encode([query]))[0]
        # ponytail: exact O(n*d) scan; use FAISS when corpus size warrants an index.
        scores = np.asarray(self.vectors) @ vector
        indices = sorted(range(len(scores)), key=lambda index: -scores[index])[:top_k]
        return tuple(self.chunks[index] for index in indices)
