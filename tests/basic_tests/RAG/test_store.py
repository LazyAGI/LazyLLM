import os
import shutil
import pytest
import tempfile
import unittest
import copy
import lazyllm
from lazyllm.tools.rag.store import (MapStore, ChromaStore, MilvusStore, OceanBaseStore,
                                     SenseCoreStore, BUILDIN_GLOBAL_META_DESC, HybridStore)
from lazyllm.tools.rag.data_type import DataType
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
data = [
    {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
     'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb1',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': None, 'answer': '', 'image_keys': []},

    {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
     'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb2',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': 'p2', 'answer': '', 'image_keys': []},

    {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
     'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
     'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {'12': 0.212890625, '23': 0.1768798828125}},
     'type': 1, 'number': 0, 'kb_id': 'kb3',
     'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
     'parent': None, 'answer': '', 'image_keys': []},
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
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The directory {directory_path} does not exist.')


class TestMapStore(unittest.TestCase):
    def setUp(self):
        self.collections = ['col_g1', 'col_g2']
        fd, self.store_dir = tempfile.mkstemp(suffix='.db')
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
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))
        store2.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        res = store2.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        store3 = MapStore(uri=self.store_dir)
        store3.connect(collections=self.collections)
        res = store3.get(collection_name=self.collections[0])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestChromaStore(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3]}, 'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},
            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1]}, 'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'p2', 'answer': '', 'image_keys': []},
            {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g1', 'content': 'test3', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1]}, 'type': 1, 'number': 0, 'kb_id': 'kb3',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},
        ]
        self.collections = ['col_g1', 'col_g2']
        self.embed_dims = {'vec_dense': 3}
        self.embed_datatypes = {'vec_dense': DataType.FLOAT_VECTOR}
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.store_dir = tempfile.mkdtemp()
        self.store = ChromaStore(uri=self.store_dir)
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
        self.collections = ['col_g1', 'col_g2']
        self.embed_dims = {'vec_dense': 3}
        self.embed_datatypes = {'vec_dense': DataType.FLOAT_VECTOR, 'vec_sparse': DataType.SPARSE_FLOAT_VECTOR}
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
        fd, self.store_dir = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.uri_standalone = ''
        self.store = MilvusStore(uri=self.store_dir, index_kwargs=self.index_kwargs)
        self.store.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                           global_metadata_desc=self.global_metadata_desc)

    def tearDown(self):
        os.remove(self.store_dir)

    def test_invalid_index_kwargs(self):
        invalid_index_kwargs = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'COSINE',
                'params': {
                    'nlist': 128,
                }
            },
            {
                'embed_key': 'vec_sparse',
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'L2',
                'params': {
                    'nlist': 128,
                }
            }]
        index_kwargs_with_no_embed_key = {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
        index_kwargs_with_one_embed_key = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            },
            {
                'index_type': 'SPARSE_INVERTED_INDEX',
                'metric_type': 'IP'
            },
        ]
        index_kwargs_with_multiple_embed_keys = [
            {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            },
            {
                'index_type': 'HNSW',
                'metric_type': 'COSINE',
            },
        ]

        test_data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},

            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'p2', 'answer': '', 'image_keys': []}
        ]
        test_data_one_embedding = [
            {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g3', 'content': 'test3', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3]},
             'type': 1, 'number': 0, 'kb_id': 'kb3',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []}
        ]

        def invalid_index_kwargs_test(invalid_index_kwargs, collections, data,
                                      embed_dims=None, embed_datatypes=None,
                                      global_metadata_desc=None):
            fd, dir = tempfile.mkstemp(suffix='.db')
            os.close(fd)
            store = None
            try:
                embed_dims = self.embed_dims if embed_dims is None else embed_dims
                embed_datatypes = self.embed_datatypes if embed_datatypes is None else embed_datatypes
                store = MilvusStore(uri=dir, index_kwargs=invalid_index_kwargs)
                store.connect(embed_dims=embed_dims, embed_datatypes=embed_datatypes,
                              global_metadata_desc=global_metadata_desc)
                if isinstance(data, dict):
                    flag = store.upsert(collections, [data])
                else:
                    flag = store.upsert(collections, data)
                return flag
            except Exception as e:
                raise e
            finally:
                if store and hasattr(store, '_client_context'):
                    try:
                        with store._client_context() as client:
                            col_list = collections if isinstance(collections, list) else [collections]
                            for col in col_list:
                                try:
                                    if client.has_collection(col):
                                        client.drop_collection(col)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                pass

        assert not invalid_index_kwargs_test(invalid_index_kwargs[0], self.collections[0], test_data[0])
        assert not invalid_index_kwargs_test(invalid_index_kwargs[1], self.collections[1], test_data[1])
        assert invalid_index_kwargs_test(index_kwargs_with_no_embed_key, self.collections[0],
                                         test_data_one_embedding, embed_datatypes={'vec_dense': DataType.FLOAT_VECTOR})
        assert invalid_index_kwargs_test(index_kwargs_with_one_embed_key, self.collections[0], test_data)
        assert invalid_index_kwargs_test(index_kwargs_with_one_embed_key, self.collections[1], test_data[1])
        with self.assertRaises(ValueError):
            invalid_index_kwargs_test(index_kwargs_with_multiple_embed_keys, self.collections[0], test_data)

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
                                query_embedding={'1563': 0.212890625, '238': 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0],
                                query_embedding={'12': 0.212890625, '23': 0.1768798828125},
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

    def test_get_massive_data(self):
        new_data_list = []
        criteria_list = []
        MASSIVE_DATA_SIZE = 20000
        for i in range(MASSIVE_DATA_SIZE):
            one_data = copy.deepcopy(data[0])
            one_data['uid'] = f'uid_{i}'
            one_data['doc_id'] = 'doc_common'
            criteria_list.append(f'uid_{i}')
            new_data_list.append(one_data)

        self.store.upsert(self.collections[0], new_data_list)

        # test client.query_iterator in get api
        res = self.store.get(collection_name=self.collections[0])
        self.assertEqual(len(res), MASSIVE_DATA_SIZE)

        SEARCH_DATA_SIZE = 9999
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': criteria_list[0:SEARCH_DATA_SIZE]})
        self.assertEqual(len(res), SEARCH_DATA_SIZE)

    def test_batch_query_legacy(self):

        with self.store._client_context() as client:
            new_data_list = []
            criteria_list = []
            for i in range(10000):
                one_data = copy.deepcopy(data[0])
                one_data['uid'] = f'uid_{i}'
                one_data['doc_id'] = 'doc_common'
                criteria_list.append(f'uid_{i}')
                new_data_list.append(one_data)

            self.store.upsert(self.collections[0], new_data_list)
            res = self.store._batch_query_legacy(client, self.collections[0], field_names=['uid'], kwargs={})
            self.assertEqual(len(res), len(new_data_list))

            filters = self.store._construct_criteria({'doc_id': 'doc_common'})
            res = self.store._batch_query_legacy(client, self.collections[0], field_names=['uid'], kwargs=filters)
            self.assertEqual(len(res), len(new_data_list))

    @pytest.mark.skip(reason=('local test for milvus standalone, please set up a milvus standalone server'
                              ' and set the uri to the server'))
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

@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestSegementStore(object):
    SEGMENTSTORE_CLASS_MAP = {
        'elasticsearch': [{
            'segment_store_type': 'elasticsearch',
            'init_kwargs': {'uris': os.getenv('ELASTICSEARCH_HOST', 'localhost:9201')},
            'is_skip': False, 'skip_reason': 'To test elasticsearch store, please set up a elasticsearch server'}],
        'opensearch': [{
            'segment_store_type': 'opensearch',
            'init_kwargs': {'uris': os.getenv('OPENSEARCH_HOST', 'localhost:9200'),
                            'client_kwargs': {
                                'user': os.getenv('OPENSEARCH_USER', 'admin'),
                                'password': os.getenv('OPENSEARCH_INITIAL_ADMIN_PASSWORD'),
                                'verify_certs': False}},
                                'user': os.getenv('OPENSEARCH_USER', 'admin'),
                                'password': os.getenv('OPENSEARCH_INITIAL_ADMIN_PASSWORD'),
                                'verify_certs': False}},
            'is_skip': False, 'skip_reason': 'To test opensearch store, please set up a opensearch server'}],
    }

    @pytest.fixture(scope='class')
    @pytest.fixture(scope='class')
    def setUP(self, request):
        collections = ['col_g1', 'col_g2']
        data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'g1', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'vec_dense': [0.1, 0.2, 0.3], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},

            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'g2', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'p2', 'answer': '', 'image_keys': []},

            {'uid': 'uid3', 'doc_id': 'doc3', 'group': 'g3', 'content': 'test3', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc3', RAG_KB_ID: 'kb3'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {'12': 0.212890625, '23': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb3',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},

            {'uid': 'uid4', 'doc_id': 'doc4', 'group': 'g4', 'content': 'test4', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc4', RAG_KB_ID: 'kb4'},
             'embedding': {'vec_dense': [0.3, 0.2, 0.1], 'vec_sparse': {'12': 0.212890625, '23': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb4',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},
        ]
        params = request.param if hasattr(request, 'param') else {}
        segment_store_type = params.get('segment_store_type')
        segment_store_init_kwargs = params.get('init_kwargs')
        is_skip = params.get('is_skip')
        skip_reason = params.get('skip_reason')
        if is_skip:
            pytest.skip(skip_reason)
        cls = getattr(lazyllm.store, segment_store_type, None)
        store = cls(**segment_store_init_kwargs)
        store.connect()
        request.cls.store = store
        request.cls.params = params
        request.cls.collections = collections
        request.cls.segment_store_type = segment_store_type
        request.cls.data = data
        return True

    @pytest.fixture()
    def upsert(self):
        self.store.upsert(self.collections[0], [self.data[0]])
        res = self.store.get(collection_name=self.collections[0])
        assert len(res) == 1, f'upsert {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[0].get('uid'), f'upsert {self.segment_store_type} failed'
        self.store.upsert(self.collections[0], [self.data[1]])
        res = self.store.get(collection_name=self.collections[0])
        assert len(res) == 2, f'upsert {self.segment_store_type} failed'
        self.store.upsert(self.collections[1], [self.data[2]])
        self.store.upsert(self.collections[1], [self.data[3]])
        return True

    @pytest.fixture()
    def get_segments_by_collection(self):
        res = self.store.get(collection_name=self.collections[0])
        assert len(res) == 2, f'get segments by collection {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1])
        assert len(res) == 2, f'get segments by collection {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def get_segments_by_kb_id(self):
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb1'})
        assert len(res) == 1, f'get segments by kb_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[0].get('uid'), f'get segments by kb_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_KB_ID: 'kb2'})
        assert len(res) == 1, f'get segments by kb_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[1].get('uid'), f'get segments by kb_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb3'})
        assert len(res) == 1, f'get segments by kb_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[2].get('uid'), f'get segments by kb_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_KB_ID: 'kb4'})
        assert len(res) == 1, f'get segments by kb_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[3].get('uid'), f'get segments by kb_id {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def get_segments_by_uid(self):
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid1']})
        assert len(res) == 1, f'get segments by uid {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[0].get('uid'), f'get segments by uid {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[0], criteria={'uid': ['uid3']})
        assert len(res) == 0, f'get segments by uid {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid3']})
        assert len(res) == 1, f'get segments by uid {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[2].get('uid'), f'get segments by uid {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={'uid': ['uid4']})
        assert len(res) == 1, f'get segments by uid {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[3].get('uid'), f'get segments by uid {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[3].get('uid'), f'get segments by uid {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def get_segments_by_doc_id(self):
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc1']})
        assert len(res) == 1, f'get segments by doc_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[0].get('uid'), f'get segments by doc_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[0], criteria={RAG_DOC_ID: ['doc4']})
        assert len(res) == 0, f'get segments by doc_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_DOC_ID: ['doc3']})
        assert len(res) == 1, f'get segments by doc_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[2].get('uid'), f'get segments by doc_id {self.segment_store_type} failed'
        res = self.store.get(collection_name=self.collections[1], criteria={RAG_DOC_ID: ['doc2']})
        assert len(res) == 0, f'get segments by doc_id {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def search(self):
        res = self.store.search(collection_name=self.collections[0], query='test')
        assert len(res) == 2, f'search {self.segment_store_type} failed'
        res = self.store.search(collection_name=self.collections[1])
        assert len(res) == 2, f'search {self.segment_store_type} failed'
        res = self.store.search(collection_name=self.collections[1], query='test3', topk=1)
        assert len(res) == 1, f'search {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[2].get('uid'), f'search {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def search_with_filters(self):
        res = self.store.search(collection_name=self.collections[0], query='test2', topk=1, filters={'group': 'g3'})
        assert len(res) == 0, f'search {self.segment_store_type} failed'
        res = self.store.search(collection_name=self.collections[1], query='test3', topk=1, filters={'group': 'g3'})
        assert len(res) == 1, f'search {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[2].get('uid'), f'search {self.segment_store_type} failed'
        res = self.store.search(collection_name=self.collections[1], query='test', filters={'group': ['g3', 'g4']})
        assert len(res) == 2, f'search {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def delete_segments_by_collection(self):
        self.store.delete(self.collections[0])
        res = self.store.get(collection_name=self.collections[0])
        assert len(res) == 0, f'delete {self.segment_store_type} {self.collections[0]} failed'
        return True

    @pytest.fixture()
    def delete_segments_by_kb_id(self):
        self.store.delete(self.collections[1], criteria={RAG_KB_ID: 'kb3'})
        res = self.store.get(collection_name=self.collections[1])
        assert len(res) == 1, f'delete segments by kb_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[3].get('uid'), f'delete segments by kb_id {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def delete_segments_by_uid(self):
        self.store.delete(self.collections[1], criteria={'uid': ['uid4']})
        res = self.store.get(collection_name=self.collections[1])
        assert len(res) == 0, f'delete segments by uid {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def delete_segments_by_doc_id(self):
        self.store.delete(self.collections[0], criteria={RAG_DOC_ID: ['doc2']})
        res = self.store.get(collection_name=self.collections[0])
        assert len(res) == 1, f'delete segments by doc_id {self.segment_store_type} failed'
        assert res[0].get('uid') == self.data[0].get('uid'), f'delete segments by doc_id \
            {self.segment_store_type} failed'
        return True

    @pytest.fixture()
    def tearDown(self):
        for collection in self.collections:
            try:
                self.store.delete(collection)
            except Exception as e:
                print(f'delete {self.segment_store_type} {collection} failed: {e}')
        return True

    def round_order(type: str, step: int):
        '''
        set the order of the test case for different store type

        type: store type
        step:  current step
        '''
        type_map = {
            'elasticsearch': 1,
            'opensearch': 2,
        }
        order_value = type_map[type] * 100 + step
        return pytest.mark.order(order_value)

    @round_order('elasticsearch', 0)
    @pytest.mark.parametrize('setUP', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_setUp(self, setUP):
        assert setUP

    @round_order('elasticsearch', 1)
    @pytest.mark.parametrize('upsert', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_upsert(self, upsert):
        assert upsert

    @round_order('elasticsearch', 2)
    @pytest.mark.parametrize('get_segments_by_collection', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_get_segments_by_collection(self, get_segments_by_collection):
        assert get_segments_by_collection

    @round_order('elasticsearch', 3)
    @pytest.mark.parametrize('get_segments_by_kb_id', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_get_segments_by_kb_id(self, get_segments_by_kb_id):
        assert get_segments_by_kb_id

    @round_order('elasticsearch', 4)
    @pytest.mark.parametrize('get_segments_by_uid', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_get_segments_by_uid(self, get_segments_by_uid):
        assert get_segments_by_uid

    @round_order('elasticsearch', 5)
    @pytest.mark.parametrize('get_segments_by_doc_id', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_get_segments_by_doc_id(self, get_segments_by_doc_id):
        assert get_segments_by_doc_id

    @round_order('elasticsearch', 6)
    @pytest.mark.parametrize('search', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_search(self, search):
        assert search

    @round_order('elasticsearch', 7)
    @pytest.mark.parametrize('search_with_filters', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_search_with_filters(self, search_with_filters):
        assert search_with_filters

    @round_order('elasticsearch', 8)
    @pytest.mark.parametrize('delete_segments_by_kb_id', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_delete_segments_by_kb_id(self, delete_segments_by_kb_id):
        assert delete_segments_by_kb_id

    @round_order('elasticsearch', 9)
    @pytest.mark.parametrize('delete_segments_by_uid', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_delete_segments_by_uid(self, delete_segments_by_uid):
        assert delete_segments_by_uid

    @round_order('elasticsearch', 10)
    @pytest.mark.parametrize('delete_segments_by_doc_id', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_delete_segments_by_doc_id(self, delete_segments_by_doc_id):
        assert delete_segments_by_doc_id

    @round_order('elasticsearch', 11)
    @pytest.mark.parametrize('delete_segments_by_collection', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_delete_segments_by_collection(self, delete_segments_by_collection):
        assert delete_segments_by_collection

    @round_order('elasticsearch', 12)
    @pytest.mark.parametrize('tearDown', SEGMENTSTORE_CLASS_MAP['elasticsearch'], indirect=True)
    def test_es_tearDown(self, tearDown):
        assert tearDown

    @round_order('opensearch', 0)
    @pytest.mark.parametrize('setUP', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_setUp(self, setUP):
        assert setUP

    @round_order('opensearch', 1)
    @pytest.mark.parametrize('upsert', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_upsert(self, upsert):
        assert upsert

    @round_order('opensearch', 2)
    @pytest.mark.parametrize('get_segments_by_collection', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_get_segments_by_collection(self, get_segments_by_collection):
        assert get_segments_by_collection

    @round_order('opensearch', 3)
    @pytest.mark.parametrize('get_segments_by_kb_id', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_get_segments_by_kb_id(self, get_segments_by_kb_id):
        assert get_segments_by_kb_id

    @round_order('opensearch', 4)
    @pytest.mark.parametrize('get_segments_by_uid', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_get_segments_by_uid(self, get_segments_by_uid):
        assert get_segments_by_uid

    @round_order('opensearch', 5)
    @pytest.mark.parametrize('get_segments_by_doc_id', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_get_segments_by_doc_id(self, get_segments_by_doc_id):
        assert get_segments_by_doc_id

    @round_order('opensearch', 6)
    @pytest.mark.parametrize('search', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_search(self, search):
        assert search

    @round_order('opensearch', 7)
    @pytest.mark.parametrize('search_with_filters', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_search_with_filters(self, search_with_filters):
        assert search_with_filters

    @round_order('opensearch', 8)
    @pytest.mark.parametrize('delete_segments_by_kb_id', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_delete_segments_by_kb_id(self, delete_segments_by_kb_id):
        assert delete_segments_by_kb_id

    @round_order('opensearch', 9)
    @pytest.mark.parametrize('delete_segments_by_uid', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_delete_segments_by_uid(self, delete_segments_by_uid):
        assert delete_segments_by_uid

    @round_order('opensearch', 10)
    @pytest.mark.parametrize('delete_segments_by_doc_id', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_delete_segments_by_doc_id(self, delete_segments_by_doc_id):
        assert delete_segments_by_doc_id

    @round_order('opensearch', 11)
    @pytest.mark.parametrize('delete_segments_by_collection', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_delete_segments_by_collection(self, delete_segments_by_collection):
        assert delete_segments_by_collection

    @round_order('opensearch', 12)
    @pytest.mark.parametrize('tearDown', SEGMENTSTORE_CLASS_MAP['opensearch'], indirect=True)
    def test_os_tearDown(self, tearDown):
        assert tearDown

@pytest.mark.skip(reason='To test sensecore store, please set up a sensecore rag-store server')
class TestSenseCoreStore(unittest.TestCase):
    def setUp(self):
        # sensecore store need kb_id when get or delete
        self.collections = ['col_block', 'col_line']
        self.data = [
            {'uid': 'uid1', 'doc_id': 'doc1', 'group': 'block', 'content': 'test1', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc1', RAG_KB_ID: 'kb1'},
             'embedding': {'bge_m3_dense': [0.1, 0.2, 0.3],
                           'bge_m3_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb1',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': None, 'answer': '', 'image_keys': []},

            {'uid': 'uid2', 'doc_id': 'doc2', 'group': 'line', 'content': 'test2', 'meta': {},
             'global_meta': {RAG_DOC_ID: 'doc2', RAG_KB_ID: 'kb2'},
             'embedding': {'bge_m3_dense': [0.3, 0.2, 0.1],
                           'bge_m3_sparse': {'1563': 0.212890625, '238': 0.1768798828125}},
             'type': 1, 'number': 0, 'kb_id': 'kb2',
             'excluded_embed_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'excluded_llm_metadata_keys': ['file_size', 'file_name', 'file_type'],
             'parent': 'uid1', 'answer': '', 'image_keys': []},
        ]
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.uri = ''
        self.s3_config = {
            'endpoint_url': os.getenv('RAG_S3_ENDPOINT', ''),
            'access_key': os.getenv('RAG_S3_ACCESS_KEY', ''),
            'secret_access_key': os.getenv('RAG_S3_SECRET_KEY', ''),
            'bucket_name': os.getenv('RAG_S3_BUCKET', 'rag-data'),
            'use_minio': os.getenv('RAG_S3_USE_MINIO', 'true').lower() == 'true',
        }
        self.image_url_config = {
            'access_key': os.getenv('RAG_IMAGE_URL_ACCESS_KEY', ''),
            'secret_access_key': os.getenv('RAG_IMAGE_URL_SECRET_KEY', ''),
            'endpoint_url': os.getenv('RAG_IMAGE_URL_ENDPOINT', ''),
            'bucket_name': os.getenv('RAG_IMAGE_URL_BUCKET', 'lazyjfs')
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
        res = self.store.search(collection_name=self.collections[0], query='test1',
                                embed_key='bge_m3_dense', topk=1,
                                filters={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        print(f'res: {res}')
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0],
                                query='test1', embed_key='bge_m3_sparse', topk=1,
                                filters={RAG_KB_ID: self.data[0].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query='test2',
                                embed_key='bge_m3_dense', topk=1,
                                filters={RAG_KB_ID: self.data[1].get('kb_id')})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), self.data[1].get('uid'))

@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
@pytest.mark.skip(reason='Oceanbase is not installed')
class TestOceanBaseStore(unittest.TestCase):
    def setUp(self):
        self.collections = ['col_g1', 'col_g2']
        self.embed_dims = {'vec_dense': 3}
        self.embed_datatypes = {'vec_dense': DataType.FLOAT_VECTOR, 'vec_sparse': DataType.SPARSE_FLOAT_VECTOR}
        self.global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        self.index_kwargs = [
            {
                'embed_key': 'vec_dense',
                'index_type': 'FLAT',
                'metric_type': 'COSINE',
            },
            {
                'embed_key': 'vec_sparse',
                'index_type': 'HNSW',
                'metric_type': 'L2',
            }
        ]
        self.store = OceanBaseStore(uri='127.0.0.1:2881', user='root@test', password='',
                                    db_name='test', index_kwargs=self.index_kwargs)
        self.store.connect(embed_dims=self.embed_dims, embed_datatypes=self.embed_datatypes,
                           global_metadata_desc=self.global_metadata_desc)

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
        res = self.store.search(collection_name=self.collections[0], query_embedding=[0.3, 0.2, 0.1],
                                embed_key='vec_dense', topk=1)
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

@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestHybridStore(unittest.TestCase):
    def setUp(self):
        self.collections = ['col_g1', 'col_g2']
        self.embed_dims = {'vec_dense': 3}
        self.embed_datatypes = {'vec_dense': DataType.FLOAT_VECTOR, 'vec_sparse': DataType.SPARSE_FLOAT_VECTOR}
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
        fd, self.store_dir = tempfile.mkstemp(suffix='.db')
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
        res = self.store.search(collection_name=self.collections[0], query='test1',
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query='test1',
                                query_embedding={'1563': 0.212890625, '238': 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query='test3',
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query='test3',
                                query_embedding={'12': 0.212890625, '23': 0.1768798828125},
                                embed_key='vec_sparse', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        res = self.store.search(collection_name=self.collections[0], query='test3',
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=5)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].get('uid'), data[2].get('uid'))
        self.assertEqual(res[1].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query='test2',
                                query_embedding=[0.3, 0.2, 0.1], embed_key='vec_dense', topk=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[1].get('uid'))

    def test_search_with_filters(self):
        self.store.upsert(self.collections[0], [data[0], data[2]])
        self.store.upsert(self.collections[1], [data[1]])
        res = self.store.search(collection_name=self.collections[0], query='test1',
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=2,
                                filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].get('uid'), data[0].get('uid'))
        res = self.store.search(collection_name=self.collections[1], query='test2',
                                query_embedding=[0.1, 0.2, 0.3], embed_key='vec_dense', topk=1,
                                filters={RAG_KB_ID: ['kb1']})
        self.assertEqual(len(res), 0)
