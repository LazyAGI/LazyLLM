import os
import tempfile
import threading
from lazyllm.tools.rag.parsing_service import DocumentProcessor
from lazyllm.tools.rag.transform import RecursiveSplitter
from lazyllm import Document, TrainableModule, LOG


def run():
    fd, milvus_store_dir = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    fd, segment_store_dir = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        milvus_store_conf = {
            'segment_store': {
                'type': 'map',
                'kwargs': {
                    'uri': segment_store_dir,
                }
            },
            'vector_store': {
                'type': 'milvus',
                'kwargs': {
                    'uri': os.getenv('MILVUS_URI', 'http://localhost:19530'),
                    'db_name': 'test_doc_123',
                    'index_kwargs': {
                        'index_type': 'IVF_FLAT',
                        'metric_type': 'COSINE',
                        'params': {
                            'nlist': 128,
                        }
                    }
                },
            }
        }

        documents = Document(
            name='doc_example',
            dataset_path=None,
            embed=TrainableModule('bge-m3'),
            server=9977,
            manager=DocumentProcessor(url='http://0.0.0.0:9966'),
            store_conf=milvus_store_conf
        )
        documents.create_node_group(
            name='sentences',
            transform=RecursiveSplitter(chunk_size=30, overlap=10, separators=[',', '.', ' ', ''])
        )
        documents.activate_groups(['sentences'])
        documents.start()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            LOG.info('\n>> Ctrl+C pressed, stopping service...')
    finally:
        try:
            os.remove(milvus_store_dir)
            os.remove(segment_store_dir)
        except Exception:
            pass

run()
