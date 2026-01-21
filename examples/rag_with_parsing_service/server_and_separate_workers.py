import os
import tempfile
import threading
import lazyllm
from lazyllm.tools.rag.parsing_service import DocumentProcessor, DocumentProcessorWorker


# callback func example
def post_func_example(task_id: str, task_status: str, error_code: str = None, error_msg: str = None):
    record = {
        'task_id': task_id,
        'task_status': task_status,
        'error_code': error_code,
        'error_msg': error_msg,
    }
    lazyllm.LOG.info(f'[callback example] record: {record}')
    return True

def run():
    fd, db_dir = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        db_config = {
            'db_type': 'sqlite',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'db_name': db_dir,
        }

        server = DocumentProcessor(port=9966, db_config=db_config, num_workers=0,
                                   post_func=post_func_example)
        server.start()

        # NOTE: db_config should be the same as server.db_config
        worker = DocumentProcessorWorker(db_config=db_config, num_workers=4, port=28888)
        worker.start()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            lazyllm.LOG.info('\n>> Ctrl+C pressed, stopping service...')
    finally:
        try:
            os.remove(db_dir)
        except Exception:
            pass

run()
