import pytest
from concurrent.futures import ThreadPoolExecutor

from lazyllm.tools.rag.store import MapStore, SegmentStore, SegmentStoreConflictError


def test_sqlite_facade_create_scope_search_and_increment(tmp_path):
    store = SegmentStore({'type': 'SQLiteStore', 'kwargs': {'db_path': str(tmp_path / 'segments.db')}})
    row = {
        'id': 'ep-1',
        'content': '用户 决定 segmentstore',
        'metadata': {'user_id': 'user-1', 'summary': '用户决定', 'hit_count': 0},
    }
    assert store.create('episodes', [row]) is True
    with pytest.raises(SegmentStoreConflictError):
        store.create('episodes', [row])
    assert store.get('episodes', {'id': 'ep-1', 'user_id': 'user-2'}) == []
    assert store.search('episodes', 'segmentstore', filters={'user_id': 'user-1'})
    assert store.patch('episodes', {'id': 'ep-1', 'user_id': 'user-1'},
                       inc_fields={'hit_count': 1}) == 1
    stored = store.get('episodes', {'id': 'ep-1', 'user_id': 'user-1'})[0]
    assert stored['metadata']['hit_count'] == 1


def test_map_facade_enforces_tenant_filter():
    store = SegmentStore(MapStore())
    store.create('episodes', [
        {'id': 'ep-1', 'content': 'segment store decision',
         'metadata': {'user_id': 'user-1', 'hit_count': 0}},
        {'id': 'ep-2', 'content': 'segment store decision',
         'metadata': {'user_id': 'user-2', 'hit_count': 0}},
    ])

    result = store.search('episodes', 'segment store', filters={'user_id': 'user-1'})

    assert [item['id'] for item in result] == ['ep-1']


def test_sqlite_atomic_increment_is_not_lost(tmp_path):
    store = SegmentStore({'type': 'SQLiteStore', 'kwargs': {'db_path': str(tmp_path / 'atomic.db')}})
    store.create('episodes', [{
        'id': 'ep-1', 'content': 'event',
        'metadata': {'user_id': 'user-1', 'hit_count': 0},
    }])

    def increment_many(_):
        for _ in range(25):
            assert store.patch('episodes', {'id': 'ep-1', 'user_id': 'user-1'},
                               inc_fields={'hit_count': 1}) == 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment_many, range(8)))

    stored = store.get('episodes', {'id': 'ep-1', 'user_id': 'user-1'})[0]
    assert stored['metadata']['hit_count'] == 200
