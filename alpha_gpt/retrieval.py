"""Scope hierarchical retrieval before searching with the paper's official Faiss dependency."""

from collections.abc import Awaitable, Callable, Sequence

from .agent import Document, Query, Stage


class FaissRetriever:
    """Index each (stage, path) bucket separately using caller-supplied embeddings.

    Buckets map to (documents, embedding rows) in matching order. Embeddings and
    queries must share a model and dimension. This uses squared L2 distance, as in
    Faiss's official getting-started example. Normalize embeddings before passing
    them here if cosine ordering is desired. Install faiss-cpu to use this adapter.
    """

    def __init__(
        self,
        buckets: dict[
            tuple[Stage, tuple[str, ...]], tuple[Sequence[Document], Sequence[Sequence[float]]]
        ],
        embed: Callable[[str], Awaitable[Sequence[float]]],
    ):
        import faiss
        import numpy as np

        self.embed = embed
        self.indexes = {}
        for key, (documents, vectors) in buckets.items():
            if not documents:
                continue
            rows = np.ascontiguousarray(vectors, dtype="float32")
            index = faiss.IndexFlatL2(rows.shape[1])
            index.add(rows)
            self.indexes[key] = (tuple(documents), index)

    async def __call__(self, query: Query) -> tuple[Document, ...]:
        import numpy as np

        bucket = self.indexes.get((query.stage, query.path))
        if bucket is None or query.limit == 0:
            return ()
        documents, index = bucket
        vector = np.ascontiguousarray([await self.embed(query.text)], dtype="float32")
        _, ids = index.search(vector, min(query.limit, len(documents)))
        return tuple(documents[i] for i in ids[0] if i >= 0)
