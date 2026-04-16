# -*- coding: utf-8 -*-
"""
ChromaDB Memory Integration for Bantu-OS.

Provides session-scoped persistent memory using ChromaDB as the vector store
and a pluggable embeddings provider.

Classes
-------
ChromaDBMemory
    Main interface: add_memory(), search_memory(), clear_memory().

Usage
-----
    from bantu_os.memory.chromadb_integration import ChromaDBMemory
    from bantu_os.memory.embeddings.openai import OpenAIEmbeddingsProvider

    embeddings = OpenAIEmbeddingsProvider(api_key="sk-...")
    memory = ChromaDBMemory(embeddings_provider=embeddings)

    # Store something for a session
    await memory.add_memory("session-abc", "User likes pizza", {"user_id": "u1"})

    # Retrieve
    results = await memory.search_memory("session-abc", "food preferences", top_k=3)

    # Clear session data
    await memory.clear_memory("session-abc")
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# ChromaDB is optional — fallback gracefully if not installed
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:  # pragma: no cover
    HAS_CHROMADB = False
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore

from bantu_os.memory.embeddings.base import EmbeddingsProvider
from bantu_os.memory.chroma_store import ChromaStore


class ChromaDBMemory:
    """Session-scoped memory backed by ChromaDB.

    Each session_id maps to a set of ChromaDB metadata-filtered records,
    enabling strict isolation between concurrent sessions.

    Parameters
    ----------
    embeddings_provider : EmbeddingsProvider
        Pluggable text->vector provider (e.g. OpenAIEmbeddingsProvider).
    collection_name : str
        ChromaDB collection name (default "bantu_memory").
    persist_path : str
        Directory for ChromaDB persistence (default "./bantu_os_data/chromadb").
    top_k : int
        Default number of results returned by search_memory().
    """

    def __init__(
        self,
        embeddings_provider: Optional[EmbeddingsProvider] = None,
        collection_name: str = "bantu_memory",
        persist_path: str = "./bantu_os_data/chromadb",
        top_k: int = 3,
    ) -> None:
        self._embeddings: Optional[EmbeddingsProvider] = embeddings_provider
        self._top_k = top_k
        self._store = ChromaStore(
            path=persist_path,
            collection=collection_name,
            distance_fn="cosine",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        session_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Embed ``text`` and store it under ``session_id``.

        Parameters
        ----------
        session_id : str
            Unique session identifier used for filtering during search.
        text : str
            Text content to embed and store.
        metadata : dict, optional
            Additional key/value metadata to attach to this memory entry.

        Returns
        -------
        str
            The internal record UID assigned by ChromaDB.

        Raises
        ------
        RuntimeError
            If no embeddings provider has been configured.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "No embeddings provider configured. "
                "Set one at construction or via set_embeddings_provider()."
            )

        embedding = await self._embeddings.embed([text])
        if embedding.ndim == 2:
            embedding = embedding[0]

        meta: Dict[str, Any] = (metadata or {}).copy()
        meta["session_id"] = session_id

        uid = self._store.add(
            embedding=embedding.tolist(),
            text=text,
            metadata=meta,
        )
        return uid

    async def search_memory(
        self,
        session_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories in ``session_id`` matching ``query``.

        Parameters
        ----------
        session_id : str
            Session to search within.
        query : str
            Natural-language query string to embed and compare.
        top_k : int, optional
            Maximum results to return (defaults to constructor's ``top_k``).

        Returns
        -------
        list[dict]
            Each dict contains at least ``text``, ``metadata``, ``similarity``.
            Returns an empty list if no matches or ChromaDB is unavailable.

        Raises
        ------
        RuntimeError
            If no embeddings provider has been configured.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "No embeddings provider configured. "
                "Set one at construction or via set_embeddings_provider()."
            )

        k = top_k if top_k is not None else self._top_k

        vec = await self._embeddings.embed([query])
        if vec.ndim == 2:
            vec = vec[0]

        results = self._store.query(
            query_embedding=vec.tolist(),
            top_k=k,
            filter_meta={"session_id": session_id},
        )
        return results

    async def clear_memory(self, session_id: str) -> int:
        """Delete all memory records belonging to ``session_id``.

        ChromaDB's delete API does not return a count, so we return ``-1``
        when the store is unavailable and ``0`` when deletion succeeds.

        Parameters
        ----------
        session_id : str
            Session whose memories should be removed.

        Returns
        -------
        int
            Number of records deleted (or ``-1`` if unavailable).
        """
        if self._store._coll is None:
            return -1

        try:
            self._store._coll.delete(where={"session_id": session_id})
            return 0
        except Exception:
            return -1

    def set_embeddings_provider(self, provider: EmbeddingsProvider) -> None:
        """Swap the embeddings provider after construction."""
        self._embeddings = provider

    @property
    def embeddings(self) -> Optional[EmbeddingsProvider]:
        """The currently configured embeddings provider, if any."""
        return self._embeddings
