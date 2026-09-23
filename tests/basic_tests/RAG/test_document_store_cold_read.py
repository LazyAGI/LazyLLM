from unittest.mock import MagicMock

import pytest

from lazyllm.tools.rag.store import HybridStore, MapStore
from lazyllm.tools.rag.store.document_store import _DocumentStore
from lazyllm.tools.rag.store.segment.sqlite_store import SQLiteStore


@pytest.mark.parametrize('paginated', [False, True])
def test_cold_hybrid_read_initializes_vectors_only_when_requested(tmp_path, paginated):
    segments = SQLiteStore(str(tmp_path / 'segments.db'))
    vectors = MapStore()
    vectors.connect = MagicMock(wraps=vectors.connect)

    def read_vectors(*args, **kwargs):
        vectors.connect.assert_called_once()
        return [{'uid': 'n1', 'embedding': {'dense': [1.0, 2.0]}}]

    vectors.get = MagicMock(side_effect=read_vectors)
    store = _DocumentStore(HybridStore(segments, vectors))
    store.activate_group('chunks')
    store.seg_impl
    try:
        assert segments.upsert(store._gen_collection_name('chunks'), [
            {'uid': 'n1', 'doc_id': 'd1', 'content': 'saved text', 'group': 'chunks', 'number': 1},
        ])
        kwargs = {'sort_by_number': True, 'return_total': True, 'limit': 10} if paginated else {}
        result = store.get_segments(group='chunks', include_embeddings=False, **kwargs)
        rows = result[0] if paginated else result
        assert rows[0]['content'] == 'saved text'
        vectors.connect.assert_not_called()
        vectors.get.assert_not_called()
        result = store.get_segments(group='chunks', **kwargs)
        rows = result[0] if paginated else result
        assert rows[0]['embedding']['dense'] == [1.0, 2.0]
        vectors.get.assert_called_once()
    finally:
        segments._open_conn().close()
