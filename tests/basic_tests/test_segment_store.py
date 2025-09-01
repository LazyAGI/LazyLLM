import pytest
from lazyllm.tools.rag.store import (OpenSearchStore, ElasticSearchStore)
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID

collections = ["col_g1", "col_g2"]
SEGMENTSTORE_CLASS_MAP = [
    {
        'segment_store_type': "elasticsearch", 'segment_store_cls': ElasticSearchStore,
        'segment_store_uri': 'localhost:9200', 'client_kwargs': {}, 'collections': collections},
    {
        'segment_store_type': 'opensearch', 'segment_store_cls': OpenSearchStore,
        'segment_store_uri': '', 'client_kwargs': {}, 'collections': collections}
]

stores = {}
for segment_store in SEGMENTSTORE_CLASS_MAP:
    if segment_store['segment_store_uri']:  # Skip if URI is empty
        try:
            segment_store_cls = segment_store['segment_store_cls']
            stores[segment_store['segment_store_type']] = segment_store_cls(segment_store['segment_store_uri'],
                                                                            client_kwargs=segment_store["client_kwargs"])
            stores[segment_store['segment_store_type']].connect()
            print(f"Successfully connected to {segment_store['segment_store_type']}")
        except Exception as e:
            raise ValueError(f"Failed to connect to {segment_store['segment_store_type']}: {e}")

# Create a filtered list with only valid stores
SEGMENTSTORE_TEST_INPUTS = []
for segment_store in SEGMENTSTORE_CLASS_MAP:
    if segment_store["segment_store_uri"] and segment_store["segment_store_type"] in stores:
        segment_store["store"] = stores[segment_store["segment_store_type"]]
        SEGMENTSTORE_TEST_INPUTS.append(segment_store)
        print(f"Added {segment_store['segment_store_type']} to test inputs")

data = [
    {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
     'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb1',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': None, 'answer': "", 'image_keys': []},

    {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
     'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb2',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': 'p2', 'answer': "", 'image_keys': []},

    {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
     'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {"12": 0.212890625, "23": 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb3',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': None, 'answer': "", 'image_keys': []},
]

@pytest.fixture()
def setUp(request):
    params = request.param if hasattr(request, 'param') else {}
    segment_store_type = params.get('segment_store_type')
    segment_store_cls = params.get('segment_store_cls')
    client_kwargs = params.get('client_kwargs')
    segment_store_uri = params.get('segment_store_uri')
    if not segment_store_uri:
        pytest.skip(f"no uri provided for {segment_store_type}")
    store = segment_store_cls(segment_store_uri, client_kwargs=client_kwargs)
    store.connect()
    return True

@pytest.fixture()
def tearDown(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        for collection in collections:
            try:
                store.delete(collection)
            except Exception as e:
                print(f"delete {store_type} {collection} failed: {e}")
    return True


@pytest.fixture()
def upsert(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0]])
        res = store.get(collection_name=collections[0])
        assert len(res) == 1, f"upsert {store_type} failed"
        assert res[0].get('uid') == data[0].get('uid'), f"upsert {store_type} failed"
    return True

@pytest.fixture()
def delete_segments_by_collection(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[1], [data[1]])
        store.delete(collections[0])
        res = store.get(collection_name=collections[0])
        assert len(res) == 0, f"delete {store_type} {collections[0]} failed"
        res = store.get(collection_name=collections[1])
        assert len(res) == 1, f"delete {store_type} {collections[1]} failed"
        assert res[0].get('uid'), data[1].get('uid')
    return True

@pytest.fixture()
def delete_segments_by_kb_id(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.delete(collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = store.get(collection_name=collections[0])
        assert len(res) == 1, f"delete segments by kb_id {store_type} failed"
        assert res[0].get('uid') == data[2].get('uid'), f"delete segments by kb_id {store_type} failed"
        store.delete(collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = store.get(collection_name=collections[0])
        assert len(res) == 0, f"delete segments by kb_id {store_type} failed"
    return True

@pytest.fixture()
def delete_segments_by_uid(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[0], [data[0], data[2]])
        store.delete(collections[0], criteria={'uid': ['uid1']})
        res = store.get(collection_name=collections[0])
        assert len(res) == 1, f"delete segments by uid {store_type} failed"
        assert res[0].get('uid') == data[2].get('uid'), f"delete segments by uid {store_type} failed"
    return True

@pytest.fixture()
def delete_segments_by_doc_id(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.delete(collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = store.get(collection_name=collections[0])
        assert len(res) == 2, f"delete segments by doc_id {store_type} failed"
        assert res[0].get('uid') == data[0].get('uid'), f"delete segments by doc_id {store_type} failed"
        assert res[1].get('uid') == data[2].get('uid'), f"delete segments by doc_id {store_type} failed"
    return True


@pytest.fixture()
def get_segments_by_collection(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[1], [data[1]])
        res = store.get(collection_name=collections[0])
        assert len(res) == 2, f"get segments by collection {store_type} failed"
        res = store.get(collection_name=collections[1])
        assert len(res) == 1, f"get segments by collection {store_type} failed"
        assert res[0].get('uid') == data[1].get('uid'), f"get segments by collection {store_type} failed"
    return True

@pytest.fixture()
def get_segments_by_kb_id(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[1], [data[1]])
        res = store.get(collection_name=collections[0], criteria={RAG_KB_ID: 'kb1'})
        assert len(res) == 1, f"get segments by kb_id {store_type} failed"
        assert res[0].get('uid'), data[0].get('uid')
        res = store.get(collection_name=collections[0], criteria={RAG_KB_ID: 'kb3'})
        assert len(res) == 1, f"get segments by kb_id {store_type} failed"
        assert res[0].get('uid') == data[2].get('uid'), f"get segments by kb_id {store_type} failed"
        res = store.get(collection_name=collections[0], criteria={RAG_KB_ID: 'kb2'})
        assert len(res) == 0, f"get segments by kb_id {store_type} failed"
        res = store.get(collection_name=collections[1], criteria={RAG_KB_ID: 'kb2'})
        assert len(res) == 1, f"get segments by kb_id {store_type} failed"
        assert res[0].get('uid') == data[1].get('uid'), f"get segments by kb_id {store_type} failed"
    return True

@pytest.fixture()
def get_segments_by_uid(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[1], [data[1]])
        res = store.get(collection_name=collections[0], criteria={'uid': ['uid1']})
        assert len(res) == 1, f"get segments by uid {store_type} failed"
        assert res[0].get('uid') == data[0].get('uid'), f"get segments by uid {store_type} failed"
        res = store.get(collection_name=collections[0], criteria={'uid': ['uid3']})
        assert len(res) == 1, f"get segments by uid {store_type} failed"
        assert res[0].get('uid') == data[2].get('uid'), f"get segments by uid {store_type} failed"
        res = store.get(collection_name=collections[0], criteria={'uid': ['uid2']})
        assert len(res) == 0, f"get segments by uid {store_type} failed"
        res = store.get(collection_name=collections[1], criteria={'uid': ['uid2']})
        assert len(res) == 1, f"get segments by uid {store_type} failed"
        assert res[0].get('uid') == data[1].get('uid'), f"get segments by uid {store_type} failed"
    return True

@pytest.fixture()
def get_segments_by_doc_id(request):
    params = request.param if hasattr(request, 'param') else {}
    collections = params.get('collections')
    store = params.get('store')
    store_type = params.get('segment_store_type')
    if store and collections:
        store.upsert(collections[0], [data[0], data[2]])
        store.upsert(collections[1], [data[1]])
        res = store.get(collection_name=collections[0], criteria={RAG_DOC_ID: ['doc1']})
        assert len(res) == 1, f"get segments by doc_id {store_type} failed"
        assert res[0].get('uid') == data[0].get('uid'), f"get segments by doc_id {store_type} failed"
        res = store.get(collection_name=collections[0], criteria={RAG_DOC_ID: ['doc2']})
        assert len(res) == 0, f"get segments by doc_id {store_type} failed"
        res = store.get(collection_name=collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        assert len(res) == 2, f"get segments by doc_id {store_type} failed"
    return True


class TestSegementStore(object):
    @pytest.mark.parametrize("setUp", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_setUp(self, setUp):
        res = setUp
        assert res

    @pytest.mark.parametrize("tearDown", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_tearDown(self, tearDown):
        res = tearDown
        assert res

    @pytest.mark.parametrize("upsert", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_upsert(self, upsert):
        res = upsert
        assert res

    @pytest.mark.parametrize("delete_segments_by_collection", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_collection(self, delete_segments_by_collection):
        res = delete_segments_by_collection
        assert res

    @pytest.mark.parametrize("delete_segments_by_kb_id", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_kb_id(self, delete_segments_by_kb_id):
        res = delete_segments_by_kb_id
        assert res

    @pytest.mark.parametrize("delete_segments_by_uid", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_uid(self, delete_segments_by_uid):
        res = delete_segments_by_uid
        assert res

    @pytest.mark.parametrize("delete_segments_by_doc_id", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_doc_id(self, delete_segments_by_doc_id):
        res = delete_segments_by_doc_id
        assert res

    @pytest.mark.parametrize("get_segments_by_collection", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_collection(self, get_segments_by_collection):
        res = get_segments_by_collection
        assert res

    @pytest.mark.parametrize("get_segments_by_kb_id", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_kb_id(self, get_segments_by_kb_id):
        res = get_segments_by_kb_id
        assert res

    @pytest.mark.parametrize("get_segments_by_uid", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_uid(self, get_segments_by_uid):
        res = get_segments_by_uid
        assert res

    @pytest.mark.parametrize("get_segments_by_doc_id", SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_doc_id(self, get_segments_by_doc_id):
        res = get_segments_by_doc_id
        assert res
