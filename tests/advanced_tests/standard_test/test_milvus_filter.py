import unittest
import time
import uuid
import os
import requests
import io
import json
import shutil

from lazyllm import config
from lazyllm import OnlineEmbeddingModule
from lazyllm.launcher import cleanup
from lazyllm.tools import Document, Retriever
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.tools.rag.global_metadata import GlobalMetadataDesc as DocField
from lazyllm.tools.rag import DataType

def get_milvus_store_conf(rag_dir: str, kb_group_name: str = str(uuid.uuid4())):
    milvus_db_dir = os.path.join(rag_dir, kb_group_name)
    if not os.path.exists(milvus_db_dir):
        os.makedirs(milvus_db_dir)

    milvus_store_conf = {
        'type': 'milvus',
        'kwargs': {
            'uri': os.path.join(milvus_db_dir, "milvus.db"),
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


def get_milvus_index_conf(rag_dir: str, kb_group_name: str = str(uuid.uuid4())):
    milvus_db_dir = os.path.join(rag_dir, kb_group_name)
    if not os.path.exists(milvus_db_dir):
        os.makedirs(milvus_db_dir)

    milvus_index_conf = {
        'type': 'map',
        'indices': {
            'smart_embedding_index': {
                'backend': 'milvus',
                'kwargs': {
                    'uri': os.path.join(milvus_db_dir, "milvus.db"),
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
        url = f"{manager_url}/add_files_to_group"
        if kw: url += ('?' + '&'.join([f'{k}={v}' for k, v in kw.items()]))
        return url

    files = [('files', ('test1.txt', io.BytesIO("跟合同相关的问题都是重要问题".encode("utf-8")), 'text/plain')),
             ('files', ('test2.txt', io.BytesIO("跟合同相关的问题都是非常重要的问题".encode("utf-8")), 'text/plain'))]
    data = dict(override='true', group_name=group, user_path='path',
                metadatas=json.dumps([{"department": "dpt_123"}, {"key_egs2": "value2"}]))
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

    def tearDown(self):
        shutil.rmtree(self.rag_dir)
        cleanup()

    def test_filter_by_tag(self):
        Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
        CUSTOM_DOC_FIELDS = {"department": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
        doc = Document(self.store_dir, name='law_kg', doc_fields=CUSTOM_DOC_FIELDS,
                       embed={"dense": OnlineEmbeddingModule(source="qwen")}, manager=True,
                       store_conf=get_milvus_store_conf(self.doc_dir, 'law_kg'))
        retriever = Retriever(doc, group_name="sentences", topk=5, embed_keys=['dense'])
        doc.start()
        time.sleep(5)

        doc_manager_url = doc._manager.url.rsplit('/', 1)[0]
        do_upload(doc_manager_url, 'law_kg')
        time.sleep(20)
        query = "合同问题"
        nodes = retriever(query, filters={'department': ['dpt_123']})
        assert len(nodes) == 1 and nodes[0].global_metadata["department"] == "dpt_123"

        nodes = retriever(query, filters={'department': 'dpt_123'})  # string instead of list
        assert len(nodes) == 1 and nodes[0].global_metadata["department"] == "dpt_123"

        # in case of re-run with old failing staus that will trigger reparsing, call release to clean db
        doc._manager._dlm.release()
        doc.stop()

    def test_smart_embedding_index(self):
        CUSTOM_DOC_FIELDS = {"department": DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' ')}
        Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
        doc = Document(self.index_dir, name='law_kg_smart', doc_fields=CUSTOM_DOC_FIELDS,
                       embed={"dense": OnlineEmbeddingModule(source="qwen")}, manager=True,
                       store_conf=get_milvus_index_conf(self.doc_dir, 'law_kg_smart'))
        retriever = Retriever(doc, group_name="sentences", topk=5, embed_keys=['dense'], similarity='cosine')
        retriever_bm25 = Retriever(doc, group_name="sentences", topk=5, similarity='bm25')
        doc.start()

        doc_manager_url = doc._manager.url.rsplit('/', 1)[0]
        do_upload(doc_manager_url, 'law_kg_smart')
        time.sleep(20)
        query = "合同问题"

        nodes = retriever(query, filters={'department': ['dpt_123']})
        assert len(nodes) == 1 and nodes[0].global_metadata["department"] == "dpt_123"

        nodes = retriever_bm25(query, filters={'department': ['dpt_123']})
        assert len(nodes) == 1 and nodes[0].global_metadata["department"] == "dpt_123"

        doc._manager._dlm.release()
        doc.stop()


if __name__ == "__main__":
    unittest.main()
