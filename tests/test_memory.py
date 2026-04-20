"""
Tests for bantu_os.memory.chromadb_integration.ChromaDBMemory.

These tests do NOT require real ChromaDB or a live embeddings API.
They use a mock in-process embeddings provider and mock/stub the
ChromaStore layer so tests run in any environment.
"""
import pytest

from bantu_os.memory.chromadb_integration import ChromaDBMemory
from bantu_os.memory.embeddings.base import EmbeddingsProvider


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockEmbeddingsProvider(EmbeddingsProvider):
    """Deterministic mock that returns a fixed zero vector per text."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._called_with: list[str] = []
        self._text_to_vector: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> "np.ndarray":  # type: ignore
        """Return a fixed 768-dim vector for each unique text.

        Subsequent calls with the same text return the same vector,
        which is important for search tests to be deterministic.
        """
        import numpy as np

        self._called_with.extend(texts)
        result = []
        for text in texts:
            if text not in self._text_to_vector:
                vec = [
                    (ord(c) % 256) / 256.0 for c in (text * (self.dim // len(text) + 1))[: self.dim]
                ]
                self._text_to_vector[text] = vec
            result.append(self._text_to_vector[text])

        return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: add_memory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chromadb_memory_add_and_search(monkeypatch):
    """add_memory() stores text and search_memory() retrieves it."""
    from bantu_os.memory.chroma_store import ChromaStore

    added_records: dict[str, dict] = {}
    seq_counter = [0]

    def tracking_add(self, embedding, text, metadata=None, uid=None):
        seq_counter[0] += 1
        uid = uid or f"mem_{seq_counter[0]}"
        added_records[uid] = {
            "id": uid,
            "text": text,
            "metadata": (metadata or {}).copy(),
            "distance": 0.0,
            "similarity": 1.0,
        }
        return uid

    def tracking_query(self, query_embedding, top_k=5, filter_meta=None):
        results = []
        for record in added_records.values():
            if filter_meta:
                match = all(record["metadata"].get(k) == v for k, v in filter_meta.items())
            else:
                match = True
            if match:
                results.append(record)
        return results[:top_k]

    monkeypatch.setattr(ChromaStore, "add", tracking_add, raising=False)
    monkeypatch.setattr(ChromaStore, "query", tracking_query, raising=False)

    memory = ChromaDBMemory(
        embeddings_provider=MockEmbeddingsProvider(dim=768),
    )

    uid = await memory.add_memory(
        session_id="session-1",
        text="I love sushi and ramen",
        metadata={"user": "alice"},
    )

    assert isinstance(uid, str)
    assert uid in added_records

    results = await memory.search_memory(
        session_id="session-1",
        query="Japanese food",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["text"] == "I love sushi and ramen"


@pytest.mark.asyncio
async def test_chromadb_memory_add_and_search__no_chromadb_available():
    """add_memory() still works when ChromaDB is not installed (no-op store)."""
    memory = ChromaDBMemory(
        embeddings_provider=MockEmbeddingsProvider(dim=768),
    )

    # When ChromaDB is unavailable, store._coll is None and add() still returns a UID
    uid = await memory.add_memory(
        session_id="session-offline",
        text="offline memory",
        metadata={"offline": True},
    )

    assert isinstance(uid, str)
    assert uid.startswith("mem_")


# ---------------------------------------------------------------------------
# Tests: Session Isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chromadb_memory_session_isolation(monkeypatch):
    """search_memory() for session-A never returns records from session-B."""
    from bantu_os.memory.chroma_store import ChromaStore

    stored: list[dict] = []

    def mock_add(self, embedding, text, metadata=None, uid=None):
        import time
        uid = uid or f"mem_{int(time.time() * 1000)}"
        stored.append({
            "id": uid,
            "text": text,
            "metadata": (metadata or {}).copy(),
            "distance": 0.0,
            "similarity": 1.0,
        })
        return uid

    def mock_query(self, query_embedding, top_k=5, filter_meta=None):
        results = []
        for rec in stored:
            if filter_meta:
                match = all(rec["metadata"].get(k) == v for k, v in filter_meta.items())
            else:
                match = True
            if match:
                results.append(rec)
        return results[:top_k]

    monkeypatch.setattr(ChromaStore, "add", mock_add, raising=False)
    monkeypatch.setattr(ChromaStore, "query", mock_query, raising=False)

    memory = ChromaDBMemory(embeddings_provider=MockEmbeddingsProvider(dim=768))

    await memory.add_memory("alice-session", "Alice's secret recipe", {"owner": "alice"})
    await memory.add_memory("alice-session", "Alice likes math", {"topic": "math"})
    await memory.add_memory("bob-session", "Bob's secret recipe", {"owner": "bob"})

    alice_results = await memory.search_memory("alice-session", "recipes and numbers")
    texts = {r["text"] for r in alice_results}
    assert "Alice's secret recipe" in texts
    assert "Alice likes math" in texts
    assert "Bob's secret recipe" not in texts

    bob_results = await memory.search_memory("bob-session", "secrets")
    bob_texts = {r["text"] for r in bob_results}
    assert "Bob's secret recipe" in bob_texts
    assert "Alice's secret recipe" not in bob_texts


@pytest.mark.asyncio
async def test_chromadb_memory_clear_isolates(monkeypatch):
    """clear_memory() only removes the target session's records."""
    from bantu_os.memory.chroma_store import ChromaStore

    stored_ref = [[]]

    def mock_add(self, embedding, text, metadata=None, uid=None):
        import time
        uid = uid or f"mem_{int(time.time() * 1000)}"
        stored_ref[0].append({
            "id": uid, "text": text,
            "metadata": (metadata or {}).copy(),
            "distance": 0.0, "similarity": 1.0,
        })
        return uid

    def mock_query(self, query_embedding, top_k=5, filter_meta=None):
        results = []
        for rec in stored_ref[0]:
            if filter_meta:
                match = all(rec["metadata"].get(k) == v for k, v in filter_meta.items())
            else:
                match = True
            if match:
                results.append(rec)
        return results[:top_k]

    monkeypatch.setattr(ChromaStore, "add", mock_add, raising=False)
    monkeypatch.setattr(ChromaStore, "query", mock_query, raising=False)

    memory = ChromaDBMemory(embeddings_provider=MockEmbeddingsProvider(dim=768))

    await memory.add_memory("s1", "memory-s1-1", {"session_id": "s1"})
    await memory.add_memory("s1", "memory-s1-2", {"session_id": "s1"})
    await memory.add_memory("s2", "memory-s2-1", {"session_id": "s2"})

    # clear_memory calls self._store._coll.delete(where={"session_id": session_id})
    # We patch _coll on the store instance to intercept that call
    class FakeColl:
        def delete(self, where):
            stored_ref[0] = [
                r for r in stored_ref[0]
                if not all(r["metadata"].get(k) == v for k, v in where.items())
            ]

    memory._store._coll = FakeColl()

    await memory.clear_memory("s1")

    remaining = await memory.search_memory("s1", "query")
    assert len(remaining) == 0

    other = await memory.search_memory("s2", "query")
    assert len(other) == 1
    assert other[0]["text"] == "memory-s2-1"


# ---------------------------------------------------------------------------
# Tests: Empty / Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chromadb_memory_empty_session(monkeypatch):
    """search_memory() on a session with no records returns an empty list."""
    from bantu_os.memory.chroma_store import ChromaStore

    monkeypatch.setattr(ChromaStore, "query", lambda self, *a, **k: [])

    memory = ChromaDBMemory(embeddings_provider=MockEmbeddingsProvider(dim=768))
    results = await memory.search_memory("unknown-session", "nothing here")
    assert results == []


@pytest.mark.asyncio
async def test_chromadb_memory_configurable_top_k(monkeypatch):
    """search_memory() respects the per-call top_k override."""
    from bantu_os.memory.chroma_store import ChromaStore

    called_top_k: list[int] = []

    def tracking_query(self, query_embedding, top_k=5, filter_meta=None):
        called_top_k.append(top_k)
        return []

    monkeypatch.setattr(ChromaStore, "query", tracking_query, raising=False)

    memory = ChromaDBMemory(
        embeddings_provider=MockEmbeddingsProvider(dim=768),
        top_k=10,
    )

    await memory.search_memory("test-session", "query", top_k=7)
    assert called_top_k[-1] == 7


@pytest.mark.asyncio
async def test_chromadb_memory_no_embeddings_raises(monkeypatch):
    """add_memory() and search_memory() raise RuntimeError without an embeddings provider."""
    memory = ChromaDBMemory()  # No embeddings provider set

    with pytest.raises(RuntimeError, match="embeddings provider"):
        await memory.add_memory("s1", "some text")

    with pytest.raises(RuntimeError, match="embeddings provider"):
        await memory.search_memory("s1", "query")


@pytest.mark.asyncio
async def test_chromadb_memory_set_embeddings_provider():
    """set_embeddings_provider() allows late injection of an embeddings provider."""
    memory = ChromaDBMemory()
    assert memory.embeddings is None

    provider = MockEmbeddingsProvider(dim=768)
    memory.set_embeddings_provider(provider)
    assert memory.embeddings is provider
