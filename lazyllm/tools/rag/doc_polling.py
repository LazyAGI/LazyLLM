import threading
import concurrent.futures
import time
import atexit
import itertools
from typing import List, Union, Callable

from .db import KBFileRecord, FileState, FileRecord
from .doc_impl import DocImpl

class DocPolling:
    """
    A class to manage file processing tasks using threading and concurrent futures.
    """
    def __init__(
            self, 
            kb_name: str, 
            add_files_fun: Callable[[List[str]], None], 
            delete_files_fun: Callable[[List[str]], None],
            batch_size=10, 
            max_workers=2
        ):
        self._kb_name = kb_name
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._add_files_fun = add_files_fun
        self._delete_files_fun = delete_files_fun

        self._condition = threading.Condition()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers)
        self._future_dict: dict[concurrent.futures.Future, List[str]] = {}
        atexit.register(self._shutdown)

    def _initialize_queue(self):
        """
        Initialize the task queue by fetching tasks with a specific state.
        """
        with self._condition:
            file_ids = self._get_task(file_state=FileState.PARSE_FAIL)
            if not file_ids:
                return
            
            self._delete_files_fun(file_ids)
            KBFileRecord.update(
                fun=lambda node: node.set(file_state=FileState.WAIT_PARSE),
                file_id=file_ids
            )

    def _shutdown(self):
        """
        Shutdown the executor and update the state of running tasks.
        """
        with self._condition:
            file_ids = itertools.chain(*[tasks for _, tasks in self._future_dict.items()])
            KBFileRecord.update(
                fun=lambda node: node.set(file_state=FileState.PARSE_FAIL),
                file_id=file_ids
            )

    def _get_task(self, file_state: FileState) -> List[str]:
        """
        Fetch tasks with a specific state.
        
        Args:
            file_state (FileState): The state of the files to fetch.
        
        Returns:
            List[str]: List of file names.
        """
        with self._condition:
            nodes = KBFileRecord.filter_by(
                limit=self._batch_size, 
                order_by=KBFileRecord.create_time.desc(), 
                file_state=file_state,
                kb_name=self._kb_name
            )
            
            return [node.file_id for node in nodes]

    def _get_running_task_size(self) -> int:
        """
        Get the current number of running tasks.
        
        Returns:
            int: Number of running tasks.
        """
        with self._condition:
            return len(self._future_dict)

    def process_tasks(self):
        """
        Main loop to process tasks.
        """
        self._initialize_queue()

        while True:
            with self._condition:
                self._delete_files()
                
                if self._get_running_task_size() < self._max_workers:
                    tasks = self._get_task(file_state=FileState.WAIT_PARSE)
                    if not tasks:
                        time.sleep(0.5)
                        continue

                    future = self._executor.submit(self.add_files_fun, tasks)
                    future.add_done_callback(self._done_callback)
                    self._future_dict[future] = tasks

                    KBFileRecord.update(
                        fun=lambda node: node.set(file_state=FileState.PENDING), 
                        file_id=tasks
                    )   
                else:
                    time.sleep(0.5)

    def _done_callback(self, future: concurrent.futures.Future):
        """
        Callback function to handle the completion of a future.
        
        Args:
            future (concurrent.futures.Future): The completed future.
        """
        with self._condition:
            if future.exception():
                KBFileRecord.update(
                    fun=lambda node: node.set(file_state=FileState.PARSE_FAIL), 
                    file_id=self._future_dict[future]
                )
                self._delete_files_fun(self._future_dict[future])
            else:
                KBFileRecord.update(
                    fun=lambda node: node.set(file_state=FileState.PARSED), 
                    file_id=self._future_dict[future]
                )

            self._future_dict.pop(future)

    def _delete_files(self):
        """
        Delete files that are marked for deletion.
        """
        with self._condition:
            file_ids = self._get_task(file_state=FileState.WAIT_DELETE)

            running_files = itertools.chain(*[tasks for _, tasks in self._future_dict.items()]) 
            del_file_ids = [_id for _id in file_ids if _id not in running_files]
            if not del_file_ids:
                return 
            
            self._delete_files_fun(del_file_ids)
            KBFileRecord.update(
                fun=lambda node: node.set(file_state=FileState.DELETED),
                file_id=del_file_ids
            )

    @staticmethod
    def start_polling(kb_name:str, doc_impl: DocImpl):
        """
        Start the file processing server.

        Args:
            kb_name (str): The name of the knowledge base.
            doc_impl (DocImpl): The implementation of document processing.
        """
        file_processor = DocPolling(
            kb_name=kb_name,
            add_files_fun=doc_impl.add_files,
            delete_files_fun=doc_impl.delete_files
        )

        thread = threading.Thread(target=file_processor.process_tasks)
        thread.daemon = True
        thread.start()
