import unittest
import time
import uuid
import os
import requests
import io
import json
import shutil

from lazyllm import config
from lazyllm import TrainableModule
from lazyllm.launcher import cleanup
from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc as DocField
from lazyllm.tools.rag import DataType

def get_milvus_store_conf(rag_dir: str, kb_group_name: str = ''):
    kb_group_name = kb_group_name or str(uuid.uuid4())
    milvus_db_dir = os.path.join(rag_dir, kb_group_name)
    if not os.path.exists(milvus_db_dir):
        os.makedirs(milvus_db_dir)

    # ``manager=True`` RAG uses a DocServer + Worker subprocess architecture,
    # so the vector store must be a deployed Milvus standalone (embedded
    # milvus_lite local .db is single-writer and would race across
    # processes; Document now rejects that combination explicitly).
    # Use ``MILVUS_URI`` to override the default CI service endpoint.
    # The db_name is randomized per call so each test invocation / rerun
    # lands on a fresh Milvus database and the assertions don't leak across
    # runs that share the same remote endpoint. See ``tearDown`` for the
    # matching best-effort drop.
    db_name = f'lazyllm_ci_{kb_group_name}_{uuid.uuid4().hex[:8]}'
    milvus_store_conf = {
        'type': 'milvus',
        'kwargs': {
            'uri': os.getenv('MILVUS_URI', 'http://10.119.26.205:19530'),
            'db_name': db_name,
            'index_kwargs': [
                {
                    'embed_key': 'dense',
                    'index_type': 'HNSW',
                    'metric_type': 'COSINE',
                },
            ]
        },
    }
    return milvus_store_conf


def _drop_milvus_db(store_conf):
    '''Best-effort cleanup so repeated test runs against a shared remote
    Milvus standalone don't accumulate per-run databases indefinitely.

    ``MilvusClient.drop_database`` requires the database to be empty of
    collections, so iterate and drop collections first. We connect with
    ``db_name`` set so the collection listing / drops target the right DB.
    Any pymilvus error (network down, db already gone, unsupported version)
    is swallowed -- this is a hygiene step, not a correctness guarantee.
    '''
    try:
        from pymilvus import MilvusClient
        kwargs = store_conf.get('kwargs') or {}
        uri = kwargs.get('uri')
        db_name = kwargs.get('db_name')
        if not uri or not db_name:
            return
        root = MilvusClient(uri=uri)
        if db_name not in root.list_databases():
            return
        scoped = MilvusClient(uri=uri, db_name=db_name)
        for col in scoped.list_collections():
            try:
                scoped.drop_collection(col)
            except Exception:
                pass
        root.drop_database(db_name)
    except Exception:
        pass


def get_milvus_index_conf(rag_dir: str, kb_group_name: str = str(uuid.uuid4())):  # noqa B008
    milvus_db_dir = os.path.join(rag_dir, kb_group_name)
    if not os.path.exists(milvus_db_dir):
        os.makedirs(milvus_db_dir)

    milvus_index_conf = {
        'type': 'map',
        'indices': {
            'smart_embedding_index': {
                'backend': 'milvus',
                'kwargs': {
                    'uri': os.path.join(milvus_db_dir, 'milvus.db'),
                    'index_kwargs': {
                        'index_type': 'HNSW',
                        'metric_type': 'COSINE',
                    }
                },
            },
        },
    }
    return milvus_index_conf


def do_upload(manager_url: str, group: str):
    def get_url(manager_url, **kw):
        url = f'{manager_url}/add_files_to_group'
        if kw: url += ('?' + '&'.join([f'{k}={v}' for k, v in kw.items()]))
        return url

    files = [('files', ('test1.txt', io.BytesIO('跟合同相关的问题都是重要问题'.encode('utf-8')), 'text/plain')),
             ('files', ('test2.txt', io.BytesIO('跟合同相关的问题都是非常重要的问题'.encode('utf-8')), 'text/plain'))]
    data = dict(override='true', group_name=group, user_path='path',
                metadatas=json.dumps([{'department': 'dpt_123'}, {'key_egs2': 'value2'}]))
    response = requests.post(get_url(manager_url, **data), files=files)
    assert response.status_code == 200


class TestMilvusFilter(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.expanduser(os.path.join(config['home'], 'rag_for_example_ut'))
        self.rag_dir = os.path.join(self.root_dir, 'milvus_filter')
        os.makedirs(self.rag_dir, exist_ok=True)
        self.doc_dir = os.path.join(self.rag_dir, 'docs')
        os.makedirs(self.doc_dir, exist_ok=True)
        self.store_dir = os.path.join(self.doc_dir, 'store')
        self.index_dir = os.path.join(self.doc_dir, 'index')
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        self._store_confs_to_drop = []

    def tearDown(self):
        for store_conf in self._store_confs_to_drop:
            _drop_milvus_db(store_conf)
        shutil.rmtree(self.rag_dir)
        cleanup()

    def test_filter_by_tag(self):
        Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
        CUSTOM_DOC_FIELDS = {'department': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
        store_conf = get_milvus_store_conf(self.doc_dir, 'law_kg')
        self._store_confs_to_drop.append(store_conf)
        doc = Document(self.store_dir, name='law_kg', doc_fields=CUSTOM_DOC_FIELDS,
                       embed={'dense': TrainableModule('bge-m3')}, manager=True,
                       store_conf=store_conf)
        retriever = Retriever(doc, group_name='sentences', topk=5, embed_keys=['dense'])
        doc.start()
        time.sleep(5)

        doc_manager_url = doc._manager.url.rsplit('/', 1)[0]
        do_upload(doc_manager_url, 'law_kg')
        time.sleep(20)
        query = '合同问题'
        nodes = retriever(query, filters={'department': ['dpt_123']})
        assert len(nodes) == 1 and nodes[0].global_metadata['department'] == 'dpt_123'

        nodes = retriever(query, filters={'department': 'dpt_123'})  # string instead of list
        assert len(nodes) == 1 and nodes[0].global_metadata['department'] == 'dpt_123'

        doc.stop()
