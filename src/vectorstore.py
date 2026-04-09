"""
ChromaDB vector store with HuggingFace local embeddings.

Stores semantic chunks from all PDFs so engineers can query across
the full report corpus with natural language.

Embeddings: sentence-transformers/all-MiniLM-L6-v2  (runs locally, no API key)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class DrillVectorStore:
    """
    Wraps ChromaDB for semantic search over drilling report chunks.
    Persists to disk so embeddings are computed once and reused.
    """

    def __init__(self, persist_dir: str | Path = VECTORSTORE_DIR):
        self._persist_dir = str(persist_dir)
        self._embeddings = _get_embeddings()
        self._db: Chroma | None = None

    # ── Build / update ────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add chunks to the vector store (creates or extends).
        Duplicate chunks (same source_file + chunk_index) are skipped.
        """
        if self._db is None:
            self._db = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embeddings,
                collection_name="drilling_reports",
            )

        # Generate stable IDs to avoid duplicates on re-runs
        ids = [
            f"{doc.metadata.get('source_file','unknown')}__chunk{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(documents)
        ]

        existing_ids = set(
            self._db.get(ids=ids)["ids"]
        )
        new_docs = [
            doc for doc, id_ in zip(documents, ids) if id_ not in existing_ids
        ]
        new_ids = [
            id_ for id_ in ids if id_ not in existing_ids
        ]

        if new_docs:
            self._db.add_documents(new_docs, ids=new_ids)

    def load(self) -> "DrillVectorStore":
        """Load existing persisted vector store from disk."""
        self._db = Chroma(
            persist_directory=self._persist_dir,
            embedding_function=self._embeddings,
            collection_name="drilling_reports",
        )
        return self

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve top-k most semantically similar chunks to the query.

        Args:
            query:  Natural language question or keyword.
            k:      Number of chunks to retrieve.
            filter: Optional Chroma metadata filter, e.g.
                    {"well_name": {"$eq": "78B-32"}}
        """
        self._ensure_loaded()
        kwargs: dict[str, Any] = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return self._db.similarity_search(query, **kwargs)

    def as_retriever(self, k: int = 5, filter: dict | None = None):
        """Return a LangChain-compatible retriever."""
        self._ensure_loaded()
        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self._db.as_retriever(search_kwargs=search_kwargs)

    def count(self) -> int:
        self._ensure_loaded()
        return self._db._collection.count()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._db is None:
            self.load()
