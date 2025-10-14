import os
import shutil
import uuid
import time
import tempfile
import requests
import unittest

from lazyllm.tools.rag.doc_processor import DocumentProcessor
from lazyllm import Document, Retriever


@unittest.skip("For local test")
class TestDocProcessor(unittest.TestCase):

    def setUp(self):
        self._store_config = {
            "vector_store": {
                "type": "milvus",
                "kwargs": {
                    "uri": os.getenv("MILVUS_URI", ""),
                    "db_name": os.getenv("MILVUS_DB_NAME", "lazyllm_test"),
                    "index_kwargs": [
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
                }
            },
            "segment_store": {
                'type': 'opensearch',
                'kwargs': {
                    'uris': os.getenv("OPENSEARCH_URI", ""),
                    'client_kwargs': {
                        "http_compress": True,
                        "use_ssl": True,
                        "verify_certs": False,
                        "user": os.getenv("OPENSEARCH_USER", ""),
                        "password": os.getenv("OPENSEARCH_PASSWORD", ""),
                    }
                }
            }
        }
        self._temp_dir = tempfile.mkdtemp()
        with open(os.path.join(self._temp_dir, "test.txt"), "w") as f:
            f.write("This is a test txt file for doc processor.\nThe answer is lazyllm.")
        self._file_path = os.path.join(self._temp_dir, "test.txt")
        self._dp_port = 14410
        self._document_port = 14411
        self.mock_embed = {'vec_dense': lambda x: [1.0, 2.0, 3.0],
                           'vec_sparse': lambda x: {0: 1.0, 1: 2.0, 2: 3.0}}
        self._algo_name = "test_algo"
        self.doc_processor = DocumentProcessor(port=self._dp_port)
        self.document = Document(dataset_path=None, name=self._algo_name, embed=self.mock_embed,
                                 store_conf=self._store_config, server=self._document_port, manager=self.doc_processor)
        self.document.create_node_group("line", display_name="Line Chunk", transform=lambda x: x.split("\n"),
                                        parent="CoarseChunk")
        self.document.activate_group("CoarseChunk", embed_keys=["vec_dense", "vec_sparse"])
        self.document.activate_group("line", embed_keys=["vec_dense", "vec_sparse"])
        self.document.start()

    def _upload_doc(self):
        url = f"http://localhost:{self._dp_port}/doc/add"
        data = {
            "task_id": uuid.uuid4().hex,
            "algo_id": self._algo_name,
            "file_infos": [
                {
                    "file_path": self._file_path,
                    "doc_id": "doc_test",
                    "metadata": {
                        "kb_id": "kb_test",
                        "test_meta": "test1"
                    }
                }
            ],
            "db_info": {
                "db_type": "mysql",
                "db_name": "db_test",
                "user": "user_test",
                "password": "password_test",
                "host": "host_test",
                "port": 3306,
                "table_name": "table_test"
            },
            "feedback_url": ""
        }
        try:
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                time.sleep(20)
            else:
                raise requests.exceptions.RequestException(f"Request failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.fail(f"Request failed: {e}")

    def tearDown(self):
        self.document.clear_cache()
        self.document.stop()
        time.sleep(5)
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def test_algo_list(self):
        url = f"http://localhost:{self._dp_port}/algo/list"
        try:
            response = requests.get(url, timeout=5)  # 添加超时
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("data")[0].get("algo_id"), self._algo_name)
        except requests.exceptions.RequestException as e:
            self.fail(f"Request failed: {e}")

    def test_retrieve(self):
        self._upload_doc()
        retriever = Retriever(doc=self.document, group_name="line", topk=2, embed_keys=["vec_dense"])
        nodes = retriever("What is the answer?")
        self.assertEqual(len(nodes), 2)

    def test_delete_doc(self):
        self._upload_doc()
        retriever = Retriever(doc=self.document, group_name="line", topk=2, embed_keys=["vec_dense"])
        nodes = retriever("What is the answer?")
        self.assertEqual(len(nodes), 2)
        url = f"http://localhost:{self._dp_port}/doc/delete"
        data = {"algo_id": self._algo_name, "dataset_id": "kb_test", "doc_ids": ["doc_test"]}
        response = requests.delete(url, json=data)
        self.assertEqual(response.status_code, 200)
        time.sleep(3)
        nodes = retriever("What is the answer?")
        self.assertEqual(len(nodes), 0)

    def test_update_meta(self):
        self._upload_doc()
        retriever = Retriever(doc=self.document, group_name="line", topk=2, embed_keys=["vec_dense"])
        nodes = retriever("What is the answer?")
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].global_metadata.get("test_meta"), "test1")
        url = f"http://localhost:{self._dp_port}/doc/meta/update"
        data = {
            "algo_id": self._algo_name,
            "file_infos": [
                {
                    "file_path": self._file_path,
                    "doc_id": "doc_test",
                    "metadata": {"kb_id": "kb_test", "test_meta": "test2"}
                }
            ],
            "db_info": {
                "db_type": "mysql",
                "db_name": "db_test",
                "user": "user_test",
                "password": "password_test",
                "host": "host_test",
                "port": 3306,
                "table_name": "table_test"
            }
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        time.sleep(20)
        nodes = retriever("What is the answer?")
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertEqual(node.global_metadata.get("test_meta"), "test2")
