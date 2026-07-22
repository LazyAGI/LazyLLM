import sqlite3
import threading

from concurrent.futures import ThreadPoolExecutor

import pytest

from lazyllm.tools.rag.store import (
    BUILDIN_GLOBAL_META_DESC,
    ChromaStore,
    ElasticSearchStore,
    MapStore,
    OpenSearchStore,
    SQLiteStore,
    create_segment_store,
)


def _segment(uid='ep-1', *, user_id='user-1', content='mars apple'):
    return {
        'uid': uid,
        'doc_id': uid,
        'group': 'episode',
        'content': content,
        'meta': {'summary': content},
        'global_meta': {},
        'kb_id': user_id,
        'counters': {'hit_count': 0, 'recall_count': 2},
    }


class _FakeIndices:
    def __init__(self):
        self.properties = {}
        self.mapping_body = None

    @staticmethod
    def exists(index):
        return True

    def get_mapping(self, index):
        return {index: {'mappings': {'properties': dict(self.properties)}}}

    def put_mapping(self, index, body):
        self.mapping_body = body
        self.properties.update(body['properties'])


class _FakeRemoteClient:
    def __init__(self):
        self.indices = _FakeIndices()
        self.body = None
        self.bulk_body = None
        self.bulk_response = {'errors': False, 'items': []}
        self.update_body = None

    def search(self, index, body):
        self.index = index
        self.body = body
        return {'hits': {'hits': []}}

    def bulk(self, index, body, refresh):
        self.bulk_body = body
        return self.bulk_response

    def update_by_query(self, index, body, refresh, conflicts):
        self.update_body = body
        return {'updated': 1, 'failures': []}


class _FailingIndices:
    @staticmethod
    def exists(index):
        raise ConnectionError(f'segment backend unavailable for {index}')


class _FailingRemoteClient:
    indices = _FailingIndices()


def _remote_store(store_cls):
    store = store_cls(uris=['http://localhost:9200'])
    client = _FakeRemoteClient()
    store._client = client
    store._ddl_lock = threading.Lock()
    store._global_metadata_desc = {}
    return store, client


def test_create_segment_store_returns_concrete_adapter_and_passes_instances_through(tmp_path):
    store = create_segment_store({
        'type': 'SQLiteStore',
        'kwargs': {'db_path': str(tmp_path / 'factory.db')},
    })
    existing = MapStore()

    assert isinstance(store, SQLiteStore)
    assert not hasattr(store, 'backend')
    assert create_segment_store(existing) is existing


@pytest.mark.parametrize(
    ('store_type', 'store_cls'),
    [('opensearch', OpenSearchStore), ('elasticsearch', ElasticSearchStore)],
)
def test_create_segment_store_uses_registered_persistent_adapter_names(store_type, store_cls):
    store = create_segment_store({
        'type': store_type,
        'kwargs': {'uris': ['http://localhost:9200']},
    })

    assert isinstance(store, store_cls)


def test_create_segment_store_validates_config_and_capability():
    vector_store = ChromaStore.__new__(ChromaStore)

    with pytest.raises(ValueError, match='requires "type"'):
        create_segment_store({})
    with pytest.raises(NotImplementedError, match='missing_store'):
        create_segment_store({'type': 'missing_store'})
    with pytest.raises(TypeError, match='config dict'):
        create_segment_store(object())
    with pytest.raises(ValueError, match='not segment-capable'):
        create_segment_store(vector_store)


def test_counter_capability_is_optional_for_existing_stores():
    store = MapStore()

    assert store.supports_counters is False
    with pytest.raises(NotImplementedError, match='named counters'):
        store.increment_counters('segments', {'uid': 'segment-1'}, {'hits': 1})


@pytest.mark.parametrize(
    'increments',
    [{}, {'bad-name': 1}, {'hit_count': True}, {'hit_count': 1.5}],
)
def test_counter_interface_rejects_invalid_increments(tmp_path, increments):
    store = SQLiteStore(db_path=str(tmp_path / 'invalid-counter.db'))

    with pytest.raises(ValueError):
        store.increment_counters('episodes', {'uid': 'ep-1'}, increments)


def test_sqlite_segment_store_lifecycle_with_named_counters(tmp_path):
    store = create_segment_store({
        'type': 'SQLiteStore',
        'kwargs': {'db_path': str(tmp_path / 'segments.db')},
    })
    store.connect(global_metadata_desc=BUILDIN_GLOBAL_META_DESC)
    row = _segment()

    assert store.supports_counters is True
    assert store.create('episodes', [row]) is True
    with pytest.raises(FileExistsError):
        store.create('episodes', [row])
    assert store.get(
        'episodes', {'uid': 'ep-1', 'kb_id': 'user-2'}, raise_on_error=True,
    ) == []
    assert store.search(
        'episodes', 'mars', filters={'kb_id': 'user-1'},
        query_fields=['content'], match_mode='any', raise_on_error=True,
    )
    assert store.increment_counters(
        'episodes', {'uid': 'ep-1', 'kb_id': 'user-1'},
        {'hit_count': 1, 'recall_count': 3},
    ) == 1

    stored = store.get(
        'episodes', {'uid': 'ep-1', 'kb_id': 'user-1'}, raise_on_error=True,
    )[0]
    assert stored['counters'] == {'hit_count': 1, 'recall_count': 5}
    assert store.delete('episodes', {'kb_id': 'user-2'}) is True
    assert store.get('episodes', {'uid': 'ep-1'}, raise_on_error=True)
    assert store.delete('episodes', {'kb_id': 'user-1'}) is True
    assert store.get('episodes', {'uid': 'ep-1'}, raise_on_error=True) == []


def test_sqlite_create_rolls_back_non_conflict_failures(tmp_path, monkeypatch):
    store = SQLiteStore(db_path=str(tmp_path / 'create-errors.db'))
    store.connect()

    missing_doc_id = _segment()
    missing_doc_id['doc_id'] = None
    with pytest.raises(sqlite3.IntegrityError):
        store.create('episodes', [missing_doc_id])

    monkeypatch.setattr(
        store,
        '_fts_content',
        lambda _item: (_ for _ in ()).throw(RuntimeError('FTS serialization failed')),
    )
    with pytest.raises(RuntimeError, match='FTS serialization failed'):
        store.create('episodes', [_segment()])

    assert store.get('episodes', {'uid': 'ep-1'}, raise_on_error=True) == []


def test_sqlite_adds_counters_to_existing_collection_without_data_rewrite(tmp_path):
    db_path = tmp_path / 'legacy.db'
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE episodes (
        uid TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "group" TEXT, content TEXT,
        meta TEXT, global_meta TEXT, type INTEGER, number INTEGER, kb_id TEXT,
        excluded_embed_metadata_keys TEXT, excluded_llm_metadata_keys TEXT,
        parent TEXT, answer TEXT, image_keys TEXT
    )''')
    conn.execute(
        'INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        ('ep-1', 'ep-1', 'episode', 'legacy event', '{}', '{}', 1, 0, 'user-1',
         '[]', '[]', None, '', '[]'),
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path=str(db_path))
    store.connect()

    assert store.get('episodes', {'uid': 'ep-1'}, raise_on_error=True)[0]['counters'] == {}
    assert store.increment_counters(
        'episodes', {'uid': 'ep-1'}, {'hit_count': 1},
    ) == 1
    assert store.get(
        'episodes', {'uid': 'ep-1'}, raise_on_error=True,
    )[0]['counters'] == {'hit_count': 1}


def test_sqlite_named_counter_increment_is_not_lost(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / 'atomic.db'))
    store.connect()
    store.create('episodes', [_segment(content='event')])

    def increment_many(_):
        for _ in range(25):
            assert store.increment_counters(
                'episodes', {'uid': 'ep-1', 'kb_id': 'user-1'}, {'hit_count': 1},
            ) == 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment_many, range(8)))

    stored = store.get('episodes', {'uid': 'ep-1'}, raise_on_error=True)[0]
    assert stored['counters']['hit_count'] == 200


def test_sqlite_counter_criteria_preserves_falsey_values(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / 'falsey-criteria.db'))
    store.connect()
    zero = _segment('ep-zero')
    zero['number'] = 0
    one = _segment('ep-one')
    one['number'] = 1
    store.create('episodes', [zero, one])

    assert store.increment_counters('episodes', {'number': 0}, {'hit_count': 1}) == 1
    assert store.increment_counters('episodes', {'uid': []}, {'hit_count': 1}) == 0

    rows = {row['uid']: row for row in store.get('episodes', raise_on_error=True)}
    assert rows['ep-zero']['counters']['hit_count'] == 1
    assert rows['ep-one']['counters']['hit_count'] == 0


def test_sqlite_strict_reads_propagate_backend_failure(monkeypatch, tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / 'strict.db'))
    monkeypatch.setattr(
        store, '_open_conn',
        lambda: (_ for _ in ()).throw(ConnectionError('SQLite unavailable')),
    )

    assert store.get('episodes') == []
    assert store.search('episodes', query='mars') == []
    with pytest.raises(ConnectionError, match='SQLite unavailable'):
        store.get('episodes', raise_on_error=True)
    with pytest.raises(ConnectionError, match='SQLite unavailable'):
        store.search('episodes', query='mars', raise_on_error=True)


def test_sqlite_segment_search_supports_any_terms_and_tenant_scope(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / 'any.db'))
    store.connect()
    store.create('episodes', [
        _segment('ep-both', content='mars apple'),
        _segment('ep-mars', content='mars banana'),
        _segment('ep-apple', content='venus apple'),
        _segment('ep-other', user_id='user-2', content='mars apple'),
    ])

    result = store.search(
        'episodes', 'mars apple', filters={'kb_id': 'user-1'},
        query_fields=['content'], match_mode='any', raise_on_error=True,
    )
    all_terms = store.search(
        'episodes', 'mars apple', filters={'kb_id': 'user-1'},
        query_fields=['content'], match_mode='all', raise_on_error=True,
    )

    assert {item['uid'] for item in result} == {'ep-both', 'ep-mars', 'ep-apple'}
    assert [item['uid'] for item in all_terms] == ['ep-both']


def test_sqlite_segment_search_rejects_non_content_field(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / 'invalid-search.db'))
    store.connect()

    with pytest.raises(ValueError, match='does not support query fields'):
        store.search('episodes', 'mars', query_fields=['summary'])


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
def test_remote_segment_store_adds_mapping_and_increments_named_counters(store_cls):
    store, client = _remote_store(store_cls)

    assert store.supports_counters is True
    assert store.increment_counters(
        'episodes', {'uid': 'ep-1', 'kb_id': 'user-1'}, {'hit_count': 1},
    ) == 1
    assert client.indices.mapping_body == {
        'properties': {'counters': {'type': 'object', 'dynamic': True}},
    }
    assert client.update_body['script']['params'] == {'increments': {'hit_count': 1}}
    must = client.update_body['query']['bool']['must']
    assert {'ids': {'values': ['ep-1']}} in must
    assert {
        'bool': {
            'should': [
                {'term': {'kb_id': 'user-1'}},
                {'term': {'kb_id.keyword': 'user-1'}},
            ],
            'minimum_should_match': 1,
        },
    } in must


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
def test_remote_segment_create_is_create_only(store_cls):
    store, client = _remote_store(store_cls)
    row = _segment(content='event')

    assert store.create('episodes', [row]) is True
    assert client.bulk_body[0] == {'create': {'_index': 'episodes', '_id': 'ep-1'}}

    client.bulk_response = {
        'errors': True,
        'items': [{'create': {'status': 409}}],
    }
    with pytest.raises(FileExistsError):
        store.create('episodes', [row])

    client.bulk_response = {
        'errors': True,
        'items': [
            {'create': {'status': 409}},
            {'create': {'status': 400, 'error': {'type': 'mapper_parsing_exception'}}},
        ],
    }
    with pytest.raises(RuntimeError, match='mapper_parsing_exception'):
        store.create('episodes', [row])


@pytest.mark.parametrize('store_cls', [SQLiteStore, OpenSearchStore, ElasticSearchStore])
def test_persistent_segment_create_rejects_non_dict_counters(store_cls, tmp_path):
    if store_cls is SQLiteStore:
        store = store_cls(db_path=str(tmp_path / 'invalid-initial-counters.db'))
        store.connect()
    else:
        store, _ = _remote_store(store_cls)
    row = _segment()
    row['counters'] = []

    with pytest.raises(ValueError, match='counters must be a dict'):
        store.create('episodes', [row])


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
def test_remote_counter_mapping_is_not_reapplied_when_already_present(store_cls):
    store, client = _remote_store(store_cls)
    client.indices.properties['counters'] = {'type': 'object', 'dynamic': True}

    assert store.increment_counters(
        'episodes', {'uid': 'ep-1'}, {'hit_count': 1},
    ) == 1
    assert client.indices.mapping_body is None


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
def test_remote_strict_reads_propagate_backend_failure(store_cls):
    store = store_cls(uris=['http://localhost:9200'])
    store._client = _FailingRemoteClient()

    assert store.get('episodes') == []
    assert store.search('episodes', query='mars') == []
    with pytest.raises(ConnectionError, match='segment backend unavailable'):
        store.get('episodes', raise_on_error=True)
    with pytest.raises(ConnectionError, match='segment backend unavailable'):
        store.search('episodes', query='mars', raise_on_error=True)


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
@pytest.mark.parametrize(('match_mode', 'operator'), [('any', 'or'), ('all', 'and')])
def test_remote_segment_search_uses_explicit_fields_operator_and_filter(
    store_cls, match_mode, operator,
):
    store, client = _remote_store(store_cls)

    assert store.search(
        'episodes', 'mars apple', filters={'kb_id': 'user-1'},
        query_fields=['content'], match_mode=match_mode, raise_on_error=True,
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


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
def test_remote_segment_search_preserves_default_scoring_semantics(store_cls):
    store, client = _remote_store(store_cls)

    assert store.search('episodes', 'mars', filters={'kb_id': 'user-1'}) == []

    assert client.body['query']['bool']['must'] == [
        {'multi_match': {'query': 'mars', 'fields': ['*']}},
        {
            'bool': {
                'should': [
                    {'term': {'kb_id': 'user-1'}},
                    {'term': {'kb_id.keyword': 'user-1'}},
                ],
                'minimum_should_match': 1,
            },
        },
    ]


@pytest.mark.parametrize('store_cls', [OpenSearchStore, ElasticSearchStore])
@pytest.mark.parametrize(
    ('query_fields', 'match_mode'),
    [([], None), ([''], None), ([None], None), ([123], None), (None, 'invalid')],
)
def test_remote_segment_search_rejects_invalid_explicit_options(
    store_cls, query_fields, match_mode,
):
    store = store_cls(uris=['http://localhost:9200'])

    with pytest.raises(ValueError):
        store.search(
            'episodes', 'mars', query_fields=query_fields, match_mode=match_mode,
        )
