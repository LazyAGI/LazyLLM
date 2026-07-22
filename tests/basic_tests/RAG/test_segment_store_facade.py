import pytest
from concurrent.futures import ThreadPoolExecutor

from lazyllm.tools.rag.store import (
    MapStore,
    OpenSearchStore,
    SegmentStore,
    SegmentStoreConflictError,
    SegmentStoreUnsupportedError,
    SQLiteStore,
)


class _FakeOpenSearchIndices:
    @staticmethod
    def exists(index):
        return True


class _FakeOpenSearchClient:
    indices = _FakeOpenSearchIndices()

    def search(self, index, body):
        self.index = index
        self.body = body
        return {'hits': {'hits': []}}


class _FailingOpenSearchIndices:
    @staticmethod
    def exists(index):
        raise ConnectionError(f'OpenSearch unavailable for {index}')


class _FailingOpenSearchClient:
    indices = _FailingOpenSearchIndices()


class _LegacySearchBackend:
    def __init__(self):
        self.calls = []

    def search(self, collection_name, query=None, topk=10, filters=None):
        self.calls.append({
            'collection_name': collection_name,
            'query': query,
            'topk': topk,
            'filters': filters,
        })
        return []


class _StrictReadBackend:
    def __init__(self):
        self.strict_values = []

    def get(self, collection_name, filters, *, raise_on_error=False):
        self.strict_values.append(raise_on_error)
        if raise_on_error:
            raise ConnectionError('segment backend unavailable')
        return []

    def search(self, collection_name, query=None, topk=10, filters=None, *, raise_on_error=False):
        self.strict_values.append(raise_on_error)
        if raise_on_error:
            raise ConnectionError('segment search backend unavailable')
        return []


def _facade_with_connected_backend(backend):
    store = SegmentStore.__new__(SegmentStore)
    store.backend = backend
    store._connected = True
    return store


@pytest.fixture
def opensearch_facade():
    backend = OpenSearchStore(uris=['http://localhost:9200'])
    client = _FakeOpenSearchClient()
    backend._client = client
    backend._global_metadata_desc = {}
    store = SegmentStore(backend)
    store._connected = True
    return store, client


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
    stored = store.get('episodes', {'id': 'ep-1', 'user_id': 'user-1'}, strict=True)[0]
    assert stored['metadata']['hit_count'] == 1
    assert store.delete('episodes', {'user_id': 'user-2'}) is True
    assert store.get('episodes', {'id': 'ep-1', 'user_id': 'user-1'}, strict=True)
    assert store.delete('episodes', {'user_id': 'user-1'}) is True
    assert store.get('episodes', {'id': 'ep-1', 'user_id': 'user-1'}, strict=True) == []


def test_facade_default_search_keeps_legacy_backend_signature_compatible():
    backend = _LegacySearchBackend()
    store = _facade_with_connected_backend(backend)

    assert store.search('episodes', 'mars', filters={'user_id': 'user-1'}) == []

    assert backend.calls == [{
        'collection_name': 'episodes',
        'query': 'mars',
        'topk': 10,
        'filters': {'kb_id': 'user-1'},
    }]


def test_facade_strict_get_propagates_backend_read_failure():
    backend = _StrictReadBackend()
    store = _facade_with_connected_backend(backend)

    assert store.get('episodes', {'user_id': 'user-1'}) == []
    with pytest.raises(ConnectionError, match='segment backend unavailable'):
        store.get('episodes', {'user_id': 'user-1'}, strict=True)

    assert backend.strict_values == [False, True]


def test_facade_strict_search_propagates_backend_read_failure():
    backend = _StrictReadBackend()
    store = _facade_with_connected_backend(backend)

    assert store.search('episodes', 'mars') == []
    with pytest.raises(ConnectionError, match='segment search backend unavailable'):
        store.search('episodes', 'mars', strict=True)

    assert backend.strict_values == [False, True]


def test_sqlite_strict_get_propagates_backend_read_failure(monkeypatch, tmp_path):
    backend = SQLiteStore(db_path=str(tmp_path / 'strict-read.db'))
    monkeypatch.setattr(
        backend,
        '_open_conn',
        lambda: (_ for _ in ()).throw(ConnectionError('SQLite unavailable')),
    )

    assert backend.get('episodes') == []
    with pytest.raises(ConnectionError, match='SQLite unavailable'):
        backend.get('episodes', raise_on_error=True)


def test_sqlite_strict_search_propagates_backend_read_failure(monkeypatch, tmp_path):
    backend = SQLiteStore(db_path=str(tmp_path / 'strict-search.db'))
    monkeypatch.setattr(
        backend,
        '_open_conn',
        lambda: (_ for _ in ()).throw(ConnectionError('SQLite search unavailable')),
    )

    assert backend.search('episodes', query='mars') == []
    with pytest.raises(ConnectionError, match='SQLite search unavailable'):
        backend.search('episodes', query='mars', raise_on_error=True)


def test_opensearch_strict_get_propagates_backend_read_failure():
    backend = OpenSearchStore(uris=['http://localhost:9200'])
    backend._client = _FailingOpenSearchClient()

    assert backend.get('episodes') == []
    with pytest.raises(ConnectionError, match='OpenSearch unavailable'):
        backend.get('episodes', raise_on_error=True)


def test_opensearch_strict_search_propagates_backend_read_failure():
    backend = OpenSearchStore(uris=['http://localhost:9200'])
    backend._client = _FailingOpenSearchClient()

    assert backend.search('episodes', query='mars') == []
    with pytest.raises(ConnectionError, match='OpenSearch unavailable'):
        backend.search('episodes', query='mars', raise_on_error=True)


@pytest.mark.parametrize(
    ('query_fields', 'match_mode'),
    [([], None), ([''], None), ([None], None), ([123], None), (None, 'invalid')],
)
def test_facade_rejects_invalid_explicit_search_contract(query_fields, match_mode):
    store = _facade_with_connected_backend(_LegacySearchBackend())

    with pytest.raises(ValueError):
        store.search(
            'episodes',
            'mars',
            query_fields=query_fields,
            match_mode=match_mode,
        )


def test_facade_rejects_explicit_search_contract_on_unsupported_backend():
    store = SegmentStore(MapStore())

    with pytest.raises(SegmentStoreUnsupportedError):
        store.search('episodes', 'mars apple', query_fields=['content'], match_mode='all')


def test_sqlite_facade_rejects_unsupported_query_field(tmp_path):
    store = SegmentStore({
        'type': 'SQLiteStore',
        'kwargs': {'db_path': str(tmp_path / 'unsupported-field.db')},
    })

    with pytest.raises(SegmentStoreUnsupportedError):
        store.search('episodes', 'mars', query_fields=['definitely_missing_field'])


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


def test_sqlite_facade_supports_any_term_search(tmp_path):
    store = SegmentStore({'type': 'SQLiteStore', 'kwargs': {'db_path': str(tmp_path / 'any.db')}})
    store.create('episodes', [
        {'id': 'ep-both', 'content': 'mars apple', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-mars', 'content': 'mars banana', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-apple', 'content': 'venus apple', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-other-user', 'content': 'mars apple', 'metadata': {'user_id': 'user-2'}},
    ])

    result = store.search(
        'episodes', 'mars apple', filters={'user_id': 'user-1'},
        query_fields=['content'], match_mode='any',
    )

    assert {item['id'] for item in result} == {'ep-both', 'ep-mars', 'ep-apple'}


def test_sqlite_facade_keeps_default_all_term_behavior(tmp_path):
    store = SegmentStore({'type': 'SQLiteStore', 'kwargs': {'db_path': str(tmp_path / 'default.db')}})
    store.create('episodes', [
        {'id': 'ep-both', 'content': 'mars apple', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-mars', 'content': 'mars banana', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-apple', 'content': 'venus apple', 'metadata': {'user_id': 'user-1'}},
    ])

    result = store.search('episodes', 'mars apple', filters={'user_id': 'user-1'})

    assert [item['id'] for item in result] == ['ep-both']


def test_sqlite_facade_safely_matches_all_terms(tmp_path):
    store = SegmentStore({'type': 'SQLiteStore', 'kwargs': {'db_path': str(tmp_path / 'all.db')}})
    store.create('episodes', [
        {'id': 'ep-match', 'content': 'EPTEST-A42 mars', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-identifier', 'content': 'EPTEST-A42 venus', 'metadata': {'user_id': 'user-1'}},
        {'id': 'ep-term', 'content': 'mars', 'metadata': {'user_id': 'user-1'}},
    ])

    result = store.search(
        'episodes', 'EPTEST-A42 mars', filters={'user_id': 'user-1'},
        query_fields=['content'], match_mode='all',
    )

    assert [item['id'] for item in result] == ['ep-match']


@pytest.mark.parametrize(('match_mode', 'operator'), [('any', 'or'), ('all', 'and')])
def test_opensearch_facade_scopes_text_search_to_content_and_filters(
    opensearch_facade, match_mode, operator,
):
    store, client = opensearch_facade

    assert store.search(
        'episodes', 'mars apple', filters={'user_id': 'user-1'},
        query_fields=['content'], match_mode=match_mode,
    ) == []

    bool_query = client.body['query']['bool']
    assert bool_query['must'] == [{
        'multi_match': {
            'query': 'mars apple',
            'fields': ['content'],
            'operator': operator,
        },
    }]
    assert bool_query['filter'] == [{
        'bool': {
            'should': [
                {'term': {'kb_id': 'user-1'}},
                {'term': {'kb_id.keyword': 'user-1'}},
            ],
            'minimum_should_match': 1,
        },
    }]


def test_opensearch_facade_keeps_default_search_options(opensearch_facade):
    store, client = opensearch_facade

    assert store.search('episodes', 'mars apple') == []

    assert client.body['query']['bool'] == {'must': [{
        'multi_match': {
            'query': 'mars apple',
            'fields': ['*'],
        },
    }]}


def test_opensearch_facade_keeps_default_filter_scoring_semantics(opensearch_facade):
    store, client = opensearch_facade

    assert store.search(
        'episodes', 'mars apple', filters={'user_id': 'user-1'},
    ) == []

    assert client.body['query']['bool'] == {'must': [
        {
            'multi_match': {
                'query': 'mars apple',
                'fields': ['*'],
            },
        },
        {
            'bool': {
                'should': [
                    {'term': {'kb_id': 'user-1'}},
                    {'term': {'kb_id.keyword': 'user-1'}},
                ],
                'minimum_should_match': 1,
            },
        },
    ]}


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
