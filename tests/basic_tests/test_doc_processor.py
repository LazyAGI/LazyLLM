import os
import shutil
import time
import tempfile
import requests
import pytest

from lazyllm.tools.rag.parsing_service import DocumentProcessor
from lazyllm.tools.rag.parsing_service.base import TaskStatus
from lazyllm import Document, Retriever

STATIC_STATUS = [TaskStatus.FINISHED.value, TaskStatus.FAILED.value, TaskStatus.CANCELED.value]

records = []
def post_func_sample(task_id: str, task_status: str, error_code: str = None, error_msg: str = None):
    records.append({
        'task_id': task_id,
        'task_status': task_status,
        'error_code': error_code,
        'error_msg': error_msg,
    })
    return True

@pytest.mark.skip(reason='For local test')
@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestDocProcessor(object):
    @classmethod
    def setUpClass(cls):
        fd, cls.segment_store_dir = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        cls._store_config = {
            'vector_store': {
                'type': 'milvus',
                'kwargs': {
                    'uri': os.getenv('MILVUS_URI', 'http://10.119.26.205:19530'),
                    'db_name': 'test_doc_processor',
                    'index_kwargs': [
                        {
                            'embed_key': 'vec_dense',
                            'index_type': 'IVF_FLAT',
                            'metric_type': 'COSINE',
                            'params': {
                                'nlist': 128,
                            }
                        }
                    ]
                },
            },
            'segment_store': {
                'type': 'map',
                'kwargs': {
                    'uri': cls.segment_store_dir,
                }
            }
        }
        cls._temp_dir = tempfile.mkdtemp()
        with open(os.path.join(cls._temp_dir, 'test.txt'), 'w') as f:
            f.write('This is a test txt file for doc processor.\nThe answer is lazyllm.')
        cls._file_path = os.path.join(cls._temp_dir, 'test.txt')

        cls._db_config = {
            'db_type': 'sqlite',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'db_name': os.path.join(cls._temp_dir, 'lazyllm_doc_parsing_task.db'),
        }
        cls._dp_port = 14410
        cls._document_port = 14411
        cls._embed = {'vec_dense': lambda x: [1.0, 2.0, 3.0]}
        cls._algo_name = 'test_algo'
        cls._finished_task_records = []
        cls.server = DocumentProcessor(port=cls._dp_port, db_config=cls._db_config, num_workers=1,
                                       post_func=post_func_sample)
        cls.document = Document(dataset_path=None, name=cls._algo_name, embed=cls._embed,
                                store_conf=cls._store_config, server=cls._document_port, manager=cls.server)
        cls.document.create_node_group('line', display_name='Line Chunk', transform=lambda x: x.split('\n'),
                                       parent='CoarseChunk')
        cls.document.activate_group('CoarseChunk', embed_keys=['vec_dense'])
        cls.document.activate_group('line', embed_keys=['vec_dense'])
        cls.document.start()

    @classmethod
    def tearDownClass(cls):
        cls.document.clear_cache()
        cls.document.drop_algorithm()
        cls.document.stop()
        cls.server.stop()
        time.sleep(5)
        if os.path.exists(cls.segment_store_dir):
            os.remove(cls.segment_store_dir)
        if os.path.exists(cls._temp_dir):
            shutil.rmtree(cls._temp_dir)

    @pytest.mark.order(0)
    def test_algo_list(self):
        url = f'http://localhost:{self._dp_port}/algo/list'
        try:
            response = requests.get(url, timeout=5)  # 添加超时
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get('data')[0].get('algo_id'), self._algo_name)
        except requests.exceptions.RequestException as e:
            self.fail(f'Request failed: {e}')

    @pytest.mark.order(1)
    def test_upload_doc(self):
        url = f'http://localhost:{self._dp_port}/doc/add'
        data = {
            'algo_id': self._algo_name,
            'file_infos': [
                {
                    'file_path': self._file_path,
                    'doc_id': 'doc_test',
                    'metadata': {
                        'kb_id': 'kb_test',
                        'test_meta': 'test1'
                    }
                }
            ]
        }
        try:
            response = requests.post(url, json=data)
            assert response.status_code == 200
            time.sleep(10)
            retriever = Retriever(doc=self.document, group_name='line', topk=2, embed_keys=['vec_dense'])
            nodes = retriever('What is the answer?')
            assert len(nodes) == 2
            assert nodes[0].global_metadata.get('test_meta') == 'test1'
        except requests.exceptions.RequestException as e:
            self.fail(f'Request failed: {e}')

    @pytest.mark.order(2)
    def test_reparse(self):
        with open(self._file_path, 'w') as f:
            f.write('This is a test txt file for doc processor.\nThe answer is lazyllm.\n new line')
        url = f'http://localhost:{self._dp_port}/doc/add'

        data = {
            'algo_id': self._algo_name,
            'file_infos': [
                {
                    'file_path': self._file_path,
                    'doc_id': 'doc_test',
                    'metadata': {'kb_id': 'kb_test', 'test_meta': 'test1'},
                    'reparse_group': 'all',
                }
            ]
        }
        response = requests.post(url, json=data)
        assert response.status_code == 200
        time.sleep(10)
        retriever = Retriever(doc=self.document, group_name='line', topk=3, embed_keys=['vec_dense'])
        nodes = retriever('What is the answer?')
        assert len(nodes) > 2
        assert nodes[0].global_metadata.get('test_meta') == 'test1'

    @pytest.mark.order(3)
    def test_update_meta(self):
        url = f'http://localhost:{self._dp_port}/doc/meta/update'
        data = {
            'algo_id': self._algo_name,
            'file_infos': [
                {
                    'file_path': self._file_path,
                    'doc_id': 'doc_test',
                    'metadata': {'kb_id': 'kb_test', 'test_meta': 'test2'}
                }
            ],
        }
        response = requests.post(url, json=data)
        assert response.status_code == 200
        time.sleep(10)
        retriever = Retriever(doc=self.document, group_name='line', topk=2, embed_keys=['vec_dense'])
        nodes = retriever('What is the answer?')
        assert len(nodes) == 2
        assert nodes[0].global_metadata.get('test_meta') == 'test2'

    @pytest.mark.order(4)
    def test_delete_doc(self):
        url = f'http://localhost:{self._dp_port}/doc/delete'
        data = {'algo_id': self._algo_name, 'kb_id': 'kb_test', 'doc_ids': ['doc_test']}
        response = requests.delete(url, json=data)
        assert response.status_code == 200
        time.sleep(10)
        retriever = Retriever(doc=self.document, group_name='line', topk=2, embed_keys=['vec_dense'])
        nodes = retriever('What is the answer?')
        assert len(nodes) == 0

    @pytest.mark.order(5)
    def test_cancel_task(self):
        url = f'http://localhost:{self._dp_port}/doc/add'
        data = {
            'algo_id': self._algo_name,
            'file_infos': [
                {
                    'file_path': self._file_path,
                    'doc_id': 'doc_test',
                    'metadata': {
                        'kb_id': 'kb_test',
                        'test_meta': 'test1'
                    }
                }
            ]
        }
        data_2 = {
            'algo_id': self._algo_name,
            'file_infos': [
                {
                    'file_path': self._file_path,
                    'doc_id': 'non_exist_doc_id',
                    'metadata': {
                        'kb_id': 'kb_test',
                        'test_meta': 'test1'
                    }
                }
            ]
        }
        try:
            response = requests.post(url, json=data)
            assert response.status_code == 200

            response = requests.post(url, json=data_2)
            assert response.status_code == 200
            task_id_2 = response.json().get('data').get('task_id')

            url = f'http://localhost:{self._dp_port}/doc/cancel'
            data = {'algo_id': self._algo_name, 'task_id': task_id_2}
            response = requests.post(url, json=data)
            assert response.status_code == 200

            time.sleep(10)
            retriever = Retriever(doc=self.document, group_name='line', topk=2, embed_keys=['vec_dense'])
            nodes = retriever('What is the answer?')
            assert len(nodes) == 2
        except requests.exceptions.RequestException as e:
            self.fail(f'Request failed: {e}')
