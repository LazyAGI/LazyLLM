# SegmentStore

`SegmentStore` 是面向非 `Document` 生命周期文本记录的公共存储 facade。
它接收 LazyLLM Store 配置或已有 Store 实例，统一提供 collection 名规范化、
延迟连接、`id/content/metadata` 逻辑记录、BM25 检索、严格追加和原子 patch。

```python
from lazyllm.tools.rag import SegmentStore

store = SegmentStore({
    "type": "SQLiteStore",
    "kwargs": {"db_path": "segments.db"},
})
store.create("events", [{
    "id": "event-1",
    "content": "版本 发布 完成",
    "metadata": {"user_id": "u1", "hit_count": 0},
}])
rows = store.search("events", "版本 发布", filters={"user_id": "u1"})
store.patch("events", {"id": "event-1", "user_id": "u1"},
            inc_fields={"hit_count": 1})
```

`create` 不会覆盖已有 ID。`patch` 不允许退化为“先读后写”；后端没有原子
能力时会抛出 `SegmentStoreUnsupportedError`。DocNode、Embedding、Vector、
Node Group 和 Hybrid 协调仍属于 `Document`。
