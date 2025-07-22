import os
import shutil
import pytest
import tempfile
import unittest
from lazyllm.tools.rag.store import MapStore, ChromadbStore, MilvusStore, OpenSearchStore, BUILDIN_GLOBAL_META_DESC
from lazyllm.tools.rag.data_type import DataType

data = [
    {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
     'global_meta': {'doc_id': 'doc1', 'kb_id': 'kb1'},
     'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb1',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': None, 'answer': "", 'image_keys': []},

    {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
     'global_meta': {'doc_id': 'doc2', 'kb_id': 'kb2'},
     'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {"1563": 0.212890625, "238": 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb2',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': 'p2', 'answer': "", 'image_keys': []},

    {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
     'global_meta': {'doc_id': 'doc3', 'kb_id': 'kb3'},
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
        _, self.store_dir = tempfile.mkstemp(suffix=".db")
        self.store1 = MapStore()
        self.store1.lazy_init(collections=self.collections)

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
        self.store1.delete(self.collections[0], criteria={'kb_id': 'kb1'})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        self.store1.delete(self.collections[0], criteria={'kb_id': 'kb3'})
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
        self.store1.delete(self.collections[0], criteria={'doc_id': ['doc2']})
        res = self.store1.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store1.delete(self.collections[0], criteria={'doc_id': ['doc1']})
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
        res = self.store1.get(collection_name=self.collections[0], criteria={'kb_id': 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])
        res = self.store1.get(collection_name=self.collections[0], criteria={'kb_id': 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        res = self.store1.get(collection_name=self.collections[0], criteria={'kb_id': 'kb2'})
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
        res = self.store1.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[0])
        res = self.store1.get(collection_name=self.collections[0], criteria={'doc_id': ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store1.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)

    def test_mapstore_with_uri(self):
        store2 = MapStore(uri=self.store_dir)
        store2.lazy_init(collections=self.collections)
        store2.upsert(self.collections[0], [data[0], data[2]])
        store2.upsert(self.collections[1], [data[1]])
        res = store2.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        res = store2.get(collection_name=self.collections[1])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[1])
        store2.delete(self.collections[0], criteria={'doc_id': ['doc1']})
        res = store2.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], data[2])
        store3 = MapStore(uri=self.store_dir)
        store3.lazy_init(collections=self.collections)
        res = store3.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))


class TestChromadbStore(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
             'global_meta': {'doc_id': 'doc1', 'kb_id': 'kb1'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3]}, 'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': "", 'image_keys': []},
            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
             'global_meta': {'doc_id': 'doc2', 'kb_id': 'kb2'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1]}, 'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'p2', 'answer': "", 'image_keys': []},
            {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
             'global_meta': {'doc_id': 'doc3', 'kb_id': 'kb3'},
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
        self.store = ChromadbStore(dir=self.store_dir)
        self.store.lazy_init(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
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
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb3'})
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
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc1']})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'kb_id': 'kb2'})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1', 'doc3']})
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
                                embed_key='vec_dense', topk=2, filters={'kb_id': ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1, filters={'kb_id': ['kb1']})
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
        _, self.store_dir = tempfile.mkstemp(suffix=".db")
        self.uri_standalone = "http://10.119.26.205:19530"
        self.store = MilvusStore(uri=self.store_dir, index_kwargs=self.index_kwargs)
        self.store.lazy_init(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
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
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb3'})
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
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc1']})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'kb_id': 'kb2'})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1', 'doc3']})
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
                                embed_key='vec_dense', topk=2, filters={'kb_id': ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query_embedding=[0.1, 0.2, 0.3],
                                embed_key='vec_dense', topk=1, filters={'kb_id': ['kb1']})
        self.assertEqual(len(res), 0)

    @pytest.mark.skip(reason=("local test for milvus standalone, please set up a milvus standalone server"
                              " and set the uri to the server"))
    def test_milvus_standalone(self):
        self.store1 = MilvusStore(uri=self.uri_standalone, index_kwargs=self.index_kwargs)
        self.store1.lazy_init(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                              global_metadata_desc=self.global_metadata_desc)
        self.store1.upsert(self.collections[0], [data[0], data[2]])
        self.store1.upsert(self.collections[1], [data[1]])
        res = self.store1.search(collection_name=self.collections[0], query_embedding=[0.1, 0.2, 0.3],
                                 embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestOpenSearchStore(unittest.TestCase):
    def setUp(self):
        self.collections = ["col_g1", "col_g2"]
        self.uri = "https://10.119.26.205:9200"
        self.client_kwargs = {
            "http_compress": True,
            "use_ssl": True,
            "verify_certs": False,
            "http_auth": ("admin", "Sensetime@123"),
        }
        self.store = OpenSearchStore(uris=self.uri, client_kwargs=self.client_kwargs)
        self.store.lazy_init()

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
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb1'})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.store.delete(self.collections[0], criteria={'kb_id': 'kb3'})
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
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 2)
        self.store.delete(self.collections[0], criteria={'doc_id': ['doc1']})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb1'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb3'})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'kb_id': 'kb2'})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[1], criteria={'kb_id': 'kb2'})
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
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc2']})
        self.assertEqual(len(res), 0)
        res = self.store.get(collection_name=self.collections[0], criteria={'doc_id': ['doc1', 'doc3']})
        self.assertEqual(len(res), 2)
