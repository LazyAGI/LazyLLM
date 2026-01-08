import os
import tempfile
import threading
import lazyllm
from lazyllm.tools.rag.parsing_service import DocumentProcessor
from lazyllm.tools.rag.parsing_service.base import TaskStatus


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

        server = DocumentProcessor(port=9966, db_config=db_config, num_workers=1,
                                   post_func=post_func_example)
        server.start()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\n>> Ctrl+C pressed, stopping service...")
    finally:
        try:
            os.remove(db_dir)
        except Exception:
            pass

run()