import os
import shutil
import pytest
import tempfile
import unittest
from lazyllm.tools.rag.store import (MapStore, ChromadbStore, MilvusStore, OpenSearchStore, ElasticSearchStore,
                                     SenseCoreStore, BUILDIN_GLOBAL_META_DESC, HybridStore)
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
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

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"The directory {directory_path} does not exist.")


class TestMapStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        fd, self.store_dir = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store1 = MapStore()
        self.store1.connect(collections=self.collections)

    def tearDown(self):
        os.remove(self.store_dir)

    def test_upsert(self):
        self.store1.upsert(self.collections[0], [data[0]])
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])

    def test_delete_segments_by_collection(self):
        self.store1.upsert(self.collections[0], [data[0]])
        self.store1.upsert(self.collections[1], [data[1]])
        self.store1.delete(self.collections[0])
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store1.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[1])

    def test_delete_segments_by_kb_id(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        self.store1.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])

    def test_delete_segments_by_doc_id(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store1.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])

    def test_get_segments_by_collection(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store1.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[1])

    def test_get_segments_by_kb_id(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)

    def test_get_segments_by_uid(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])
        res = self.store1.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        res = self.store1.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)

    def test_get_segments_by_doc_id(self):
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store1.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)

    def test_mapstore_with_uri(self):
        store2 = MapStore(uri=self.store_dir)
        store2.connect(collections=self.collections)
        store2.upsert(self.collections[0], [data[0], data[2]])
        store2.upsert(self.collections[1], [data[1]])
        res = store2.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = store2.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[1])
        store2.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = store2.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        store3 = MapStore(uri=self.store_dir)
        store3.connect(collections=self.collections)
        res = store3.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestChromadbStore(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3]}, 'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': "", 'image_keys': []},
            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1]}, 'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'p2', 'answer': "", 'image_keys': []},
            {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1]}, 'type': 1, 'number': 0, 'kb_id': 'kb3',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': "", 'image_keys': []},
        ]
        self.collections = ["col_g1", "col_g2"]
        self.embed_dims = {"vec_dense": 3}
        self.embed_datatypes = {"vec_dense": DataType.FLOAT_VECTOR}
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.store_dir = tempfile.mkdtemp()
        self.store = ChromadbStore(uri=self.store_dir)
        self.store.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                           global_metadata_desc=self.global_metadata_desc)

    def tearDown(self):
        clear_directory(self.store_dir)

    def test_upsert(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))

    def test_delete_segments_by_collection(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_delete_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))

    def test_get_segments_by_collection(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_get_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)

    def test_search(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        self.assertEqual(res[1].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_search_with_filters(self):
        self.store.upsert(self.collections[0], [self.data[0], self.data[2]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=2, filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1, filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 0)


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestMilvusStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        self.embed_dims = {"vec_dense": 3}
        self.embed_datatypes = {"vec_dense": DataType.FLOAT_VECTOR, "vec_sparse": DataType.SPARSE_FLOAT_VECTOR}
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.index_kwargs = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'FLAT',
                'metric_type': 'COSINE',
                'params': {
                    'nlist': 128,
                }
            },
            {
                'embed_key': 'vec_sparse',
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'IP',
                'params': {
                    'nlist': 128,
                }
            }
        ]
        fd, self.store_dir = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.uri_standalone = ""
        self.store = MilvusStore(uri=self.store_dir, index_kwargs=self.index_kwargs)
        self.store.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                           global_metadata_desc=self.global_metadata_desc)

    def tearDown(self):
        os.remove(self.store_dir)

    def test_upsert(self):
        self.store.upsert(self.collections[0], [data[0]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))

    def test_delete_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_delete_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_get_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_kb_and_doc(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1', RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)

    def test_search(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0],
                                query_embedding={"1563": 0.212890625, "238": 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0],
                                query_embedding={"12": 0.212890625, "23": 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.assertEqual(res[1].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_search_with_filters(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=2, filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1, filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 0)

    @pytest.mark.skip(reason=("local test for milvus standalone, please set up a milvus standalone server"
                              " and set the uri to the server"))
    def test_milvus_standalone(self):
        self.store1 = MilvusStore(uri=self.uri_standalone, index_kwargs=self.index_kwargs)
        self.store1.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                            global_metadata_desc=self.global_metadata_desc)
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                 embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))


@pytest.mark.skip(reason="To test open search store, please set up a open search server")
class TestOpenSearchStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        self.uri = ""
        self.client_kwargs = {}
        self.store = OpenSearchStore(uris=self.uri, client_kwargs=self.client_kwargs)
        self.store.connect()

    def tearDown(self):
        for collection in self.collections:
            self.store.delete(collection)

    def test_upsert(self):
        self.store.upsert(self.collections[0], [data[0]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))

    def test_delete_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_delete_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_get_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)


@pytest.mark.skip(reason="To test elastic search store, please set up a elastic search server")
class TestElasticSearchStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        self.uri = ""
        self.client_kwargs = {}
        self.store = ElasticSearchStore(uris=self.uri, client_kwargs=self.client_kwargs)
        self.store.connect()

    def tearDown(self):
        for collection in self.collections:
            self.store.delete(collection)

    def test_upsert(self):
        self.store.upsert(self.collections[0], [data[0]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))

    def test_delete_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_delete_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_get_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)


class TestSegementStore(object):
    collections = ["col_g1", "col_g2"]
    SEGMENTSTORE_CLASS_MAP = [
        {
            'segment_store_type': "elasticsearch", 'segment_store_cls': ElasticSearchStore,
            'segment_store_uri': '', 'client_kwargs': {}, 'collections': collections,
            'is_skip': False, 'skip_reason': 'To test elasticsearch store, please set up a elasticsearch server'},
        {
            'segment_store_type': 'opensearch', 'segment_store_cls': OpenSearchStore,
            'segment_store_uri': '', 'client_kwargs': {}, 'collections': collections,
            'is_skip': True, 'skip_reason': 'To test opensearch store, please set up a opensearch server'}
    ]

    stores = {}
    for segment_store in SEGMENTSTORE_CLASS_MAP:
        if not segment_store['is_skip']:  # Skip if is_skip is True
            try:
                segment_store_cls = segment_store['segment_store_cls']
                stores[segment_store['segment_store_type']] = segment_store_cls(
                    segment_store['segment_store_uri'],
                    client_kwargs=segment_store['client_kwargs']
                )
                stores[segment_store['segment_store_type']].connect()
                print(f"Successfully connected to {segment_store['segment_store_type']}")
            except Exception as e:
                raise ValueError(f"Failed to connect to {segment_store['segment_store_type']}: {e}")

    # Create a filtered list with only valid stores
    SEGMENTSTORE_TEST_INPUTS = []
    for segment_store in SEGMENTSTORE_CLASS_MAP:
        if segment_store['segment_store_uri'] and segment_store['segment_store_type'] in stores:
            segment_store['store'] = stores[segment_store['segment_store_type']]
        SEGMENTSTORE_TEST_INPUTS.append(segment_store)

    @pytest.fixture()
    def setUp(self, request):
        params = request.param if hasattr(request, 'param') else {}
        segment_store_cls = params.get('segment_store_cls')
        client_kwargs = params.get('client_kwargs')
        segment_store_uri = params.get('segment_store_uri')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        store = segment_store_cls(segment_store_uri, client_kwargs=client_kwargs)
        store.connect()
        return True

    @pytest.fixture()
    def tearDown(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        if store and collections:
            for collection in collections:
                try:
                    store.delete(collection)
                except Exception as e:
                    print(f"delete {store_type} {collection} failed: {e}")
        return True

    @pytest.fixture()
    def upsert(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        if store and collections:
            store.upsert(collections[0], [data[0]])
            res = store.get(collection_name=collections[0])
            assert len(res) == 1, f"upsert {store_type} failed"
            assert res[0].get('uid') == data[0].get('uid'), f"upsert {store_type} failed"
        return True

    @pytest.fixture()
    def delete_segments_by_collection(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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
    def delete_segments_by_kb_id(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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
    def delete_segments_by_uid(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        if store and collections:
            store.upsert(collections[0], [data[0], data[2]])
            store.upsert(collections[0], [data[0], data[2]])
            store.delete(collections[0], criteria={'uid': ['uid1']})
            res = store.get(collection_name=collections[0])
            assert len(res) == 1, f"delete segments by uid {store_type} failed"
            assert res[0].get('uid') == data[2].get('uid'), f"delete segments by uid {store_type} failed"
        return True

    @pytest.fixture()
    def delete_segments_by_doc_id(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        if store and collections:
            store.upsert(collections[0], [data[0], data[2]])
            store.delete(collections[0], criteria={RAG_DOC_ID: ['doc2']})
            res = store.get(collection_name=collections[0])
            assert len(res) == 2, f"delete segments by doc_id {store_type} failed"
            assert res[0].get('uid') == data[0].get('uid'), f"delete segments by doc_id {store_type} failed"
            assert res[1].get('uid') == data[2].get('uid'), f"delete segments by doc_id {store_type} failed"
        return True

    @pytest.fixture()
    def get_segments_by_collection(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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
    def get_segments_by_kb_id(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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
    def get_segments_by_uid(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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
    def get_segments_by_doc_id(self, request):
        params = request.param if hasattr(request, 'param') else {}
        collections = params.get('collections')
        store = params.get('store')
        store_type = params.get('segment_store_type')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
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

    @pytest.mark.parametrize('setUp', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_setUp(self, setUp):
        res = setUp
        assert res

    @pytest.mark.parametrize('tearDown', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_tearDown(self, tearDown):
        res = tearDown
        assert res

    @pytest.mark.parametrize('upsert', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_upsert(self, upsert):
        res = upsert
        assert res

    @pytest.mark.parametrize('delete_segments_by_collection', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_collection(self, delete_segments_by_collection):
        res = delete_segments_by_collection
        assert res

    @pytest.mark.parametrize('delete_segments_by_kb_id', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_kb_id(self, delete_segments_by_kb_id):
        res = delete_segments_by_kb_id
        assert res

    @pytest.mark.parametrize('delete_segments_by_uid', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_uid(self, delete_segments_by_uid):
        res = delete_segments_by_uid
        assert res

    @pytest.mark.parametrize('delete_segments_by_doc_id', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_delete_segments_by_doc_id(self, delete_segments_by_doc_id):
        res = delete_segments_by_doc_id
        assert res

    @pytest.mark.parametrize('get_segments_by_collection', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_collection(self, get_segments_by_collection):
        res = get_segments_by_collection
        assert res

    @pytest.mark.parametrize('get_segments_by_kb_id', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_kb_id(self, get_segments_by_kb_id):
        res = get_segments_by_kb_id
        assert res

    @pytest.mark.parametrize('get_segments_by_uid', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_uid(self, get_segments_by_uid):
        res = get_segments_by_uid
        assert res

    @pytest.mark.parametrize('get_segments_by_doc_id', SEGMENTSTORE_TEST_INPUTS, indirect=True)
    def test_get_segments_by_doc_id(self, get_segments_by_doc_id):
        res = get_segments_by_doc_id
        assert res

@pytest.mark.skip(reason="To test sensecore store, please set up a sensecore rag-store server")
class TestSenseCoreStore(unittest.TestCase):
    def setUp(self):
        # sensecore store need kb_id when get or delete
        self.collections = ["col_block", "col_line"]
        self.data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'block', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'bge_m3_dense': [0.1, 0.2, 0.3],
                           'bge_m3_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': "", 'image_keys': []},

            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'line', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'bge_m3_dense': [0.3, 0.2, 0.1],
                           'bge_m3_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'uid1', 'answer': "", 'image_keys': []},
        ]
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.uri = ""
        self.s3_config = {
            "endpoint_url": os.getenv("RAG_S3_ENDPOINT", ""),
            "access_key": os.getenv("RAG_S3_ACCESS_KEY", ""),
            "secret_access_key": os.getenv("RAG_S3_SECRET_KEY", ""),
            "bucket_name": os.getenv("RAG_S3_BUCKET", "rag-data"),
            "use_minio": os.getenv("RAG_S3_USE_MINIO", "true").lower() == "true",
        }
        self.image_url_config = {
            "access_key": os.getenv("RAG_IMAGE_URL_ACCESS_KEY", ""),
            "secret_access_key": os.getenv("RAG_IMAGE_URL_SECRET_KEY", ""),
            "endpoint_url": os.getenv("RAG_IMAGE_URL_ENDPOINT", ""),
            "bucket_name": os.getenv("RAG_IMAGE_URL_BUCKET", "lazyjfs")
        }
        self.store = SenseCoreStore(uri=self.uri, s3_config=self.s3_config, image_url_config=self.image_url_config)
        self.store.connect(global_metadata_desc=self.global_metadata_desc)

    def tearDown(self):
        try:
            self.store.delete(self.collections[0], criteria={'uid': ['uid1'], RAG_KB_ID: self.data[0].get('kb_id')})
            self.store.delete(self.collections[1], criteria={'uid': ['uid2'], RAG_KB_ID: self.data[1].get('kb_id')})
        except Exception:
            pass

    def test_upsert(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.store.delete(self.collections[0], criteria={'uid': ['uid1'], RAG_KB_ID: self.data[0].get('kb_id')})
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2'], RAG_KB_ID: self.data[0].get('kb_id')})
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1'], RAG_KB_ID: self.data[0].get('kb_id')})
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 0)

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0],
                             criteria={'uid': ['uid1'], RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0],
                             criteria={'uid': ['uid2'], RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1],
                             criteria={'uid': ['uid2'], RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0],
                             criteria={RAG_DOC_ID: ['doc1'], RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0],
                             criteria={RAG_DOC_ID: ['doc2'], RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1],
                             criteria={RAG_DOC_ID: ['doc2'], RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

    def test_search(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        self.store.upsert(self.collections[1], [self.data[1]])
        res = self.store.search(collection_name=self.collections[0], query="test1",
                                embed_key='bge_m3_dense', topk=1,
                                filters={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        print(f"res: {res}")
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0],
                                query="test1", embed_key='bge_m3_sparse', topk=1,
                                filters={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query="test2",
                                embed_key='bge_m3_dense', topk=1,
                                filters={RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestHybridStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        self.embed_dims = {"vec_dense": 3}
        self.embed_datatypes = {"vec_dense": DataType.FLOAT_VECTOR, "vec_sparse": DataType.SPARSE_FLOAT_VECTOR}
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.index_kwargs = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'FLAT',
                'metric_type': 'COSINE',
                'params': {
                    'nlist': 128,
                }
            },
            {
                'embed_key': 'vec_sparse',
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'IP',
                'params': {
                    'nlist': 128,
                }
            }
        ]
        fd, self.store_dir = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.segment_store = MapStore()
        self.vector_store = MilvusStore(uri=self.store_dir, index_kwargs=self.index_kwargs)
        self.store = HybridStore(self.segment_store, self.vector_store)
        self.store.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                           global_metadata_desc=self.global_metadata_desc, collections=self.collections)

    def tearDown(self):
        os.remove(self.store_dir)

    def test_upsert(self):
        self.store.upsert(self.collections[0], [data[0]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))

    def test_delete_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_delete_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 0)

    def test_delete_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={'uid': ['uid1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_delete_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0], )
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))

    def test_get_segments_by_collection(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = self.store.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_kb_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb2'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_uid(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid2']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_get_segments_by_doc_id(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)

    def test_search(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.search(collection_name=self.collections[0], query="test1",
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query="test1",
                                query_embedding={"1563": 0.212890625, "238": 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query="test3",
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query="test3",
                                query_embedding={"12": 0.212890625, "23": 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query="test3",
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.assertEqual(res[1].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query="test2",
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_search_with_filters(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.search(collection_name=self.collections[0], query="test1",
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=2,
                                filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query="test2",
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=1,
                                filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 0)
