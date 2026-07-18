# SegmentStore

`SegmentStore` is the public facade for persistent text segments that do not
belong to the `Document` lifecycle. It accepts a LazyLLM store configuration or
an existing store instance and exposes normalized collection names, lazy
connection, canonical `id/content/metadata` records, BM25 search, create-only
writes, and atomic patches.

```python
from lazyllm.tools.rag import SegmentStore

store = SegmentStore({
    "type": "SQLiteStore",
    "kwargs": {"db_path": "segments.db"},
})
store.create("events", [{
    "id": "event-1",
    "content": "release completed",
    "metadata": {"user_id": "u1", "hit_count": 0},
}])
rows = store.search("events", "release", filters={"user_id": "u1"})
store.patch("events", {"id": "event-1", "user_id": "u1"},
            inc_fields={"hit_count": 1})
```

`create` never overwrites an existing ID. `patch` never falls back to a
read-modify-write sequence: a backend without an atomic primitive raises
`SegmentStoreUnsupportedError`. `Document` continues to own DocNode,
embedding, vector, node-group, and hybrid coordination behavior.
