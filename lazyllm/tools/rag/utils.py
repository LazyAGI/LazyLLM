import concurrent
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (Any, Callable, Dict, Generator, List, Optional, Set, Tuple,
                    Union)
from urllib.parse import urlsplit, urlunsplit

import pydantic
import sqlalchemy
from lazyllm.thirdparty import fastapi
from filelock import FileLock
from pydantic import BaseModel
from sqlalchemy import Column, Row, bindparam, insert, select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import lazyllm
from lazyllm import config
from lazyllm.common import override
from lazyllm.common.queue import sqlite3_check_threadsafety
from lazyllm.thirdparty import tarfile

from .doc_node import DocNode
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH
from .index_base import IndexBase
from pathlib import Path

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    'max_embedding_workers',
    int,
    min(32, (os.cpu_count() or 1) + 4),
    'MAX_EMBEDDING_WORKERS',
)

config.add('default_dlmanager', str, 'sqlite', 'DEFAULT_DOCLIST_MANAGER')

def gen_docid(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()


class KBDataBase(DeclarativeBase):
    pass


class KBOperationLogs(KBDataBase):
    __tablename__ = 'operation_logs'
    id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    log = Column(sqlalchemy.Text, nullable=False)
    created_at = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False)


DocPartRow = Row
class KBDocument(KBDataBase):
    __tablename__ = 'documents'

    doc_id = Column(sqlalchemy.Text, primary_key=True)
    filename = Column(sqlalchemy.Text, nullable=False, index=True)
    path = Column(sqlalchemy.Text, nullable=False)
    created_at = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False)
    last_updated = Column(sqlalchemy.DateTime, default=sqlalchemy.func.now(), onupdate=sqlalchemy.func.now())
    meta = Column(sqlalchemy.Text, nullable=True)
    status = Column(sqlalchemy.Text, nullable=False, index=True)
    count = Column(sqlalchemy.Integer, default=0)

class KBGroup(KBDataBase):
    __tablename__ = 'document_groups'

    group_id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    group_name = Column(sqlalchemy.String, nullable=False, unique=True)

DocMetaChangedRow = Row
GroupDocPartRow = Row

class KBGroupDocuments(KBDataBase):
    __tablename__ = 'kb_group_documents'

    id = Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    doc_id = Column(sqlalchemy.String, sqlalchemy.ForeignKey('documents.doc_id'), nullable=False)
    group_name = Column(sqlalchemy.String, sqlalchemy.ForeignKey('document_groups.group_name'), nullable=False)
    status = Column(sqlalchemy.Text, nullable=True)
    log = Column(sqlalchemy.Text, nullable=True)
    need_reparse = Column(sqlalchemy.Boolean, default=False, nullable=False)
    new_meta = Column(sqlalchemy.Text, nullable=True)
    # unique constraint
    __table_args__ = (sqlalchemy.UniqueConstraint('doc_id', 'group_name', name='uq_doc_to_group'),)


class DocPathParsingResult(BaseModel):
    doc_id: str
    success: bool
    msg: str
    is_new: bool = False

class DocListManager(ABC):
    """Abstract base class for managing document lists and monitoring changes in a document directory.

Args:
    path: Path of the document directory to monitor.
    name: Name of the manager.
    enable_path_monitoring: Whether to enable path monitoring.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.rag.utils import DocListManager
    >>> manager = DocListManager(path='your_file_path/', name="test_manager", enable_path_monitoring=False)
    >>> added_docs = manager.add_files([test_file_list])
    >>> manager.enable_path_monitoring(True)
    >>> deleted = manager.delete_files([delete_file_list])
    """
    DEFAULT_GROUP_NAME = '__default__'
    __pool__ = dict()

    class Status:
        all = 'all'
        waiting = 'waiting'
        working = 'working'
        success = 'success'
        failed = 'failed'
        deleting = 'deleting'
        # deleted is no longer used
        deleted = 'deleted'

    def __init__(self, path, name, enable_path_monitoring=True):
        self._path = path
        self._name = name
        lazyllm.LOG.info(f'DocManager use file-system monitoring worker: {enable_path_monitoring}')
        self._id = hashlib.sha256(f'{name}@+@{path}'.encode()).hexdigest()
        if not os.path.isabs(path):
            raise ValueError(f'path [{path}] is not an absolute path')

        self._init_sql()
        self._delete_nonexistent_docs_on_startup()

        self._monitor_thread = threading.Thread(target=self._monitor_directory_worker)
        self._monitor_thread.daemon = True
        self._monitor_continue = True
        self._enable_path_monitoring = enable_path_monitoring
        self._init_monitor_event = threading.Event()
        if self._enable_path_monitoring:
            self._monitor_thread.start()
            self._init_monitor_event.wait()

    def _delete_nonexistent_docs_on_startup(self):
        ids = [row[0] for row in self.list_kb_group_files(details=True)
               if not Path(row[1]).exists()]
        if ids: self.delete_files(ids)

    def __new__(cls, *args, **kw):
        if cls is not DocListManager:
            return super().__new__(cls)
        return super().__new__(__class__.__pool__[config['default_dlmanager']])

    def init_tables(self) -> 'DocListManager':
        """Ensure that the default group exists in the database tables.
"""
        if not self.table_inited():
            self._init_tables()
        # in case of using after relase
        self.add_kb_group(DocListManager.DEFAULT_GROUP_NAME)
        return self

    def _monitor_directory(self) -> Set[str]:
        files_list = []
        for root, _, files in os.walk(self._path):
            files = [os.path.join(root, file_path) for file_path in files]
            files_list.extend(files)
        return set(files_list)

    # Actually it shoule be 'set_docs_status_deleting'
    def delete_files(self, file_ids: List[str]) -> List[DocPartRow]:
        """Set the knowledge base entries associated with the document to "deleting," and have each knowledge base asynchronously delete parsed results and associated records.

Args:
    file_ids (list of str): List of file IDs to delete.
"""
        document_list = self.update_file_status(file_ids, DocListManager.Status.deleting)
        self.update_kb_group(cond_file_ids=file_ids, new_status=DocListManager.Status.deleting)
        return document_list

    @abstractmethod
    def table_inited(self):
        """Checks if the database table `documents` is initialized. This method ensures thread-safety when accessing the database.
Determines whether the `documents` table exists in the database.

**Returns:**

- bool: `True` if the `documents` table exists, `False` otherwise.

Notes:
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe access to the database.
    - Establishes a connection to the SQLite database at `self._db_path` with the `check_same_thread` option.
    - Executes the SQL query: `SELECT name FROM sqlite_master WHERE type='table' AND name='documents'` to check for the table.
"""
        pass

    @abstractmethod
    def _init_tables(self): pass

    @abstractmethod
    def validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]:
        """Validates a list of file paths to ensure they are ready for processing.
This method checks whether the provided paths are new, already processed, or currently being processed. It ensures there are no conflicts in processing the documents.

Args:
    paths (List[str]): A list of file paths to validate.

**Returns:**

- Tuple[bool, str, List[bool]]: A tuple containing:
        - `bool`: `True` if all paths are valid, `False` otherwise.
        - `str`: A message indicating success or the reason for failure.
        - `List[bool]`: A list where each element corresponds to whether a path is new (`True`) or already exists (`False`).
Notes:
    - If any document is still being processed or needs reparsing, the method returns `False` with an appropriate error message.
    - The method uses a database session and thread-safe lock (`self._db_lock`) to retrieve document status information.
    - Unsafe statuses include `working` and `waiting`.

"""
        pass

    @abstractmethod
    def update_need_reparsing(self, doc_id: str, need_reparse: bool):
        """Updates the `need_reparse` status of a document in the `KBGroupDocuments` table.
This method sets the `need_reparse` flag for a specific document, optionally scoped to a given group.

Args:
    doc_id (str): The ID of the document to update.
    need_reparse (bool): The new value for the `need_reparse` flag.

Notes:
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The method commits the change to the database immediately.
"""
        pass

    @abstractmethod
    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None):
        """Lists files from the `documents` table with optional filtering, limiting, and returning details.
This method retrieves file IDs or detailed file information from the database, based on the specified filtering conditions.

Args:
    limit (Optional[int]): Maximum number of files to return. If `None`, all matching files will be returned.
    details (bool): Whether to return detailed file information (`True`) or just file IDs (`False`).
    status (Union[str, List[str]]): The status or list of statuses to include in the results. Defaults to all statuses.
    exclude_status (Optional[Union[str, List[str]]]): The status or list of statuses to exclude from the results. Defaults to `None`.

**Returns:**

- List: A list of file IDs if `details=False`, or a list of detailed file rows if `details=True`.

Notes:
    - The method constructs a query dynamically based on the provided `status` and `exclude_status` conditions.
    - A thread-safe lock (`self._db_lock`) ensures safe database access.
    - The `LIMIT` clause is applied if `limit` is specified.
"""
        pass

    @abstractmethod
    def get_docs(self, doc_ids: List[str]) -> List[KBDocument]:
        """This method retrieves document objects of type `KBDocument` from the database for the provided list of document IDs.

Args:
    doc_ids (List[str]): A list of document IDs to fetch.

**Returns:**

- List[KBDocument]: A list of `KBDocument` objects corresponding to the provided document IDs. If no documents are found, an empty list is returned.

Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - It performs a SQL join between `KBDocument` and `KBGroupDocuments` to retrieve the relevant rows.
    - After fetching, it updates the `new_meta` field of the affected rows to `None` and commits the changes to the database.
"""
        pass

    @abstractmethod
    def set_docs_new_meta(self, doc_meta: Dict[str, dict]):
        """Batch update metadata for documents.

Args:
    doc_meta (Dict[str, dict]): A dictionary mapping document IDs to their new metadata.
"""
        pass

    @abstractmethod
    def fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]:
        """List files in a specific knowledge base (KB) group with optional filters, limiting, and details.
This method retrieves files from the `kb_group_documents` table, optionally filtering by group, document status, upload status, and whether reparsing is needed.

Args:
    group (str): The name of the group to filter documents by.

**Returns:**

- List[DocMetaChangedRow]: A list of rows, where each row contains the `doc_id` and the `new_meta` field of documents with changed metadata.

Notes:
    - This method constructs a SQL query dynamically based on the provided filters.
    - Uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - If `status` or `upload_status` are provided as lists, they are processed with SQL `IN` clauses.
"""
        pass

    @abstractmethod
    def list_all_kb_group(self):
        """Lists all the knowledge base group names.

**Returns:**

- list: List of knowledge base group names.
"""
        pass

    @abstractmethod
    def add_kb_group(self, name):
        """Adds a new knowledge base group.

Args:
    name (str): Name of the group to add.
"""
        pass

    @abstractmethod
    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: Union[str, List[str]] = Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None,
                            need_reparse: Optional[bool] = False):
        """List files in a specific knowledge base group .

Args:
    group (str): The name of the KB group to filter files by. Defaults to `None` .
    limit (Optional[int]): Maximum number of files to return. If `None`, returns all matching files.
    details (bool): Whether to return detailed file information or only file IDs and paths.
    status (Union[str, List[str]]): The KB group status or list of statuses to include in the results. Defaults to all statuses.
    exclude_status (Optional[Union[str, List[str]]): The KB group status or list of statuses to exclude from the results. Defaults to `None`.
    upload_status (Union[str, List[str]]): The document upload status or list of statuses to include in the results. Defaults to all statuses.
    exclude_upload_status (Optional[Union[str, List[str]]): The document upload status or list of statuses to exclude from the results. Defaults to `None`.
    need_reparse (Optional[bool]): Whether to filter files that need reparsing or not . Defaults to `None` .

**Returns:**

- List: If `details=False`, returns a list of tuples containing `(doc_id, path)`. 
          If `details=True`, returns a list of detailed rows with additional metadata.
Notes:
    - The method first creates document records using the `_add_doc_records` helper function.
    - After the files are added, they are automatically linked to the default KB group (`DocListManager.DEFAULT_GROUP_NAME`).
    - Batch processing ensures scalability when adding a large number of files.
"""
        pass

    def add_files(
        self,
        files: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        status: Optional[str] = Status.waiting,
        batch_size: int = 64,
    ) -> List[DocPartRow]:
        """Add multiple files to the document list with optional metadata, status, and batch processing.
This method adds a list of files to the database and sets optional metadata and initial status for each file. The files are processed in batches for efficiency. After the files are added, they are automatically associated with the default knowledge base (KB) group.

Args:
    files (List[str]): A list of file paths to add to the database.
    metadatas (Optional[List[Dict[str, Any]]]): A list of metadata dictionaries corresponding to the files. If `None`, no metadata will be associated. Defaults to `None`.
    status (Optional[str]): The initial status for the added files. Defaults to `Status.waiting`.
    batch_size (int): The number of files to process in each batch. Defaults to 64.

**Returns:**

- List[DocPartRow]: A list of `DocPartRow` objects representing the added files and their associated information.

Notes:
- The method first creates document records using the helper function _add_doc_records.
- After the files are added, they are automatically linked to the default knowledge base group (DocListManager.DEFAULT_GROUP_NAME).
- Batch processing ensures good scalability when adding a large number of files.


"""
        documents = self._add_doc_records(files, metadatas, status, batch_size)
        if documents:
            self.add_files_to_kb_group([doc.doc_id for doc in documents], group=DocListManager.DEFAULT_GROUP_NAME)
        return documents

    @abstractmethod
    def _get_all_docs(self): pass

    @abstractmethod
    def _get_docs(self, to_be_added_doc_ids: List, to_be_deleted_doc_ids: List, filter_status_list: List): pass

    @abstractmethod
    def _add_doc_records(self, files: List[str], metadatas: Optional[List] = None,
                         status: Optional[str] = Status.waiting, batch_size: int = 64) -> List[DocPartRow]: pass

    @abstractmethod
    def delete_unreferenced_doc(self):
        """Delete documents marked as "deleting" and no longer referenced in the database.
This method removes documents from the database that meet the following conditions:
    1. Their status is set to `DocListManager.Status.deleting`.
    2. Their reference count (`count`) is 0.
"""
        pass

    @abstractmethod
    def get_docs_need_reparse(self, group: Optional[str] = None) -> List[KBDocument]:
        """Retrieve documents that require reparsing for a specific group.
This method fetches documents that are marked as needing reparsing (`need_reparse=True`) for the given group. Only documents with a status of `success` or `failed` are included in the results.

Args:
    group (str): The name of the group to filter documents by.

**Returns:**

- List[KBDocument]: A list of `KBDocument` objects that need reparsing.

Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The query performs a SQL `JOIN` between `KBDocument` and `KBGroupDocuments` to filter by group and reparse status.
    - Documents with `need_reparse=True` and a status of `success` or `failed` are considered for reparsing.
"""
        pass

    @abstractmethod
    def get_existing_paths_by_pattern(self, file_path: str) -> List[str]:
        """Retrieve existing document paths that match a given pattern.
This method fetches all document paths from the database that match the provided SQL `LIKE` pattern.

Args:
    pattern (str): The SQL `LIKE` pattern to filter document paths. For example, `%example%` matches paths containing the word "example".

**Returns:**

- List[str]: A list of document paths that match the given pattern. If no paths match, an empty list is returned.

Notes:
    - The method uses a thread-safe lock (`self._db_lock`) to ensure safe database access.
    - The `LIKE` operator in the SQL query is used to perform pattern matching on document paths.
"""
        pass

    @abstractmethod
    def update_file_message(self, fileid: str, **kw):
        """Updates the message for a specified file.

Args:
    fileid (str): File ID.
    **kw: Additional key-value pairs to update.
"""
        pass

    @abstractmethod
    def update_file_status(self, file_ids: List[str], status: str,
                           cond_status_list: Union[None, List[str]] = None) -> List[DocPartRow]:
        """Update the status of specified files.

Args:
    file_ids (list of str): List of file IDs whose status needs to be updated.
    status (str): Target status to set.
    cond_status_list (Union[None, List[str]]): Optional. Only update files currently in these statuses.
"""
        pass

    @abstractmethod
    def add_files_to_kb_group(self, file_ids: List[str], group: str):
        """Adds files to the specified knowledge base group.

Args:
    file_ids (list of str): List of file IDs to add.
    group (str): Name of the group to add the files to.
"""
        pass

    @abstractmethod
    def delete_files_from_kb_group(self, file_ids: List[str], group: str):
        """Deletes files from the specified knowledge base group.

Args:
    file_ids (list of str): List of file IDs to delete.
    group (str): Name of the group.
"""
        pass

    @abstractmethod
    def get_file_status(self, fileid: str):
        """Retrieves the status of a specified file.

Args:
    fileid (str): File ID.

**Returns:**

- str: The current status of the file.
"""
        pass

    @abstractmethod
    def update_kb_group(self, cond_file_ids: List[str], cond_group: Optional[str] = None,
                        cond_status_list: Optional[List[str]] = None, new_status: Optional[str] = None,
                        new_need_reparse: Optional[bool] = None) -> List[GroupDocPartRow]:
        """Updates the record of kb_group_document.

Args:
    cond_file_ids (list of str, optional): a list of file IDs to filter by, default None.
    cond_group (str, optional): a kb_group name to filter by, default None.
    cond_status_list (list of str, optional): a list of statuses to filter by, default None.
    new_status (str, optional): the new status to update to, default None
    new_need_reparse (bool, optinoal): the new need_reparse flag to update to, default None

**Returns:**

- list: updated records, list of (doc_id, group_name)
"""
        pass

    @abstractmethod
    def release(self):
        """Releases the resources of the current manager.
"""
        pass

    @property
    def enable_path_monitoring(self):
        """Enable or disable path monitoring for the document manager.
This method enables or disables the path monitoring functionality in the document manager. When enabled, a monitoring thread starts to handle path-related operations. When disabled, the thread stops and joins (waits for it to terminate).

Args:
    val (bool): Whether to enable or disable path monitoring.

Notes:
    - If `val` is `True`, path monitoring is enabled by setting `_monitor_continue` to `True` and starting the `_monitor_thread`.
    - If `val` is `False`, path monitoring is disabled by setting `_monitor_continue` to `False` and joining the `_monitor_thread` if it is running.
    - This method ensures thread-safe operation when managing the monitoring thread.
"""
        return self._enable_path_monitoring

    @enable_path_monitoring.setter
    def enable_path_monitoring(self, val: bool):
        """Enable or disable path monitoring for the document manager.
This method enables or disables the path monitoring functionality in the document manager. When enabled, a monitoring thread starts to handle path-related operations. When disabled, the thread stops and joins (waits for it to terminate).

Args:
    val (bool): Whether to enable or disable path monitoring.

Notes:
    - If `val` is `True`, path monitoring is enabled by setting `_monitor_continue` to `True` and starting the `_monitor_thread`.
    - If `val` is `False`, path monitoring is disabled by setting `_monitor_continue` to `False` and joining the `_monitor_thread` if it is running.
    - This method ensures thread-safe operation when managing the monitoring thread.
"""
        self._enable_path_monitoring = (val is True)
        if val is True:
            self._monitor_continue = True
            self._monitor_thread.start()
        else:
            self._monitor_continue = False
            if self._monitor_thread.is_alive():
                self._monitor_thread.join()

    def _monitor_directory_worker(self):
        failed_files_count = defaultdict(int)
        docs_all = self._get_all_docs()

        previous_files = set([doc.path for doc in docs_all])
        skip_files = set()
        is_first_run = True
        while self._monitor_continue:
            # 1. Scan files in the directory, find added and deleted files
            current_files = set(self._monitor_directory())
            to_be_added_files = current_files - previous_files - skip_files
            to_be_deleted_files = previous_files - current_files - skip_files
            failed_files = set()

            to_be_added_doc_ids = set([gen_docid(ele) for ele in to_be_added_files])
            to_be_deleted_doc_ids = set([gen_docid(ele) for ele in to_be_deleted_files])
            failed_doc_ids = set()
            filter_status_list = [DocListManager.Status.success,
                                  DocListManager.Status.failed, DocListManager.Status.waiting]

            docs_not_expected, docs_expected = self._get_docs(to_be_added_doc_ids,
                                                              to_be_deleted_doc_ids, filter_status_list)

            # 2. Skip new files that are already in the database
            failed_files.update([doc.path for doc in docs_not_expected])
            failed_doc_ids.update([doc.doc_id for doc in docs_not_expected])
            to_be_added_files -= failed_files
            # Actually it is add to doc with success status, then add to kb_group with waiting status
            self.add_files(list(to_be_added_files), status=DocListManager.Status.success)

            # 3. Skip deleted files that are: 1. not in the database, 2. status not success/failed/waiting
            safe_to_delete_files = set([doc.path for doc in docs_expected])
            safe_to_delete_doc_ids = set([doc.doc_id for doc in docs_expected])
            failed_doc_ids.update(to_be_deleted_doc_ids - safe_to_delete_doc_ids)
            failed_files.update(to_be_deleted_files - safe_to_delete_files)
            to_be_deleted_files = safe_to_delete_files
            to_be_deleted_doc_ids = safe_to_delete_doc_ids
            self.delete_files(list(to_be_deleted_doc_ids))

            # 4. update skip_files
            for ele in failed_files:
                failed_files_count[ele] += 1
                if failed_files_count[ele] >= 3:
                    skip_files.add(ele)
            # update previous files, while failed files will be re-processed in the next loop
            previous_files = (current_files | to_be_added_files) - to_be_deleted_files
            if is_first_run:
                self._init_monitor_event.set()
            is_first_run = False
            time.sleep(10)
        lazyllm.LOG.warning('END MONITORING')

    def __del__(self):
        self.enable_path_monitoring = False


class SqliteDocListManager(DocListManager):
    """SQLite-based document manager for persistent local file storage, status tracking, and metadata management.

This class inherits from DocListManager and uses a SQLite backend to store document records. It is suitable for managing locally identified documents with support for inserting, querying, updating, and filtering based on status. Optional file path monitoring is also supported.

Args:
    path (str): Directory path to store the database.
    name (str): Name of the SQLite database file (without path).
    enable_path_monitoring (bool): Whether to enable path monitoring. Defaults to True.


Examples:
    >>> from lazyllm.tools.rag.utils import SqliteDocListManager
    >>> manager = SqliteDocListManager(path="./data", name="docs.sqlite")
    >>> manager.insert({"uid": "doc_001", "name": "example.txt", "status": "ready"})
    >>> print(manager.get("doc_001"))
    >>> files = manager.list_files(limit=5, details=True)
    >>> print(files)
    """
    def __init__(self, path, name, enable_path_monitoring=True):
        super().__init__(path, name, enable_path_monitoring)

    def _init_sql(self):
        root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
        os.makedirs(root_dir, exist_ok=True)
        self._db_path = os.path.join(root_dir, f'.lazyllm_dlmanager.{self._id}.db')
        self._db_lock = FileLock(self._db_path + '.lock')
        # ensure that this connection is not used in another thread when sqlite3 is not threadsafe
        self._check_same_thread = not sqlite3_check_threadsafety()
        self._engine = sqlalchemy.create_engine(
            f'sqlite:///{self._db_path}?check_same_thread={self._check_same_thread}'
        )
        self._Session = sessionmaker(bind=self._engine)
        self.init_tables()

    def _init_tables(self):
        KBDataBase.metadata.create_all(bind=self._engine)

    def table_inited(self):
        """Checks whether the "documents" table has been initialized in the database.

The method queries the sqlite_master metadata table to verify if the "documents" table exists.

**Returns:**

- bool: True if the "documents" table exists, False otherwise.
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'documents\'')
            return cursor.fetchone() is not None

    @staticmethod
    def get_status_cond_and_params(status: Union[str, List[str]],
                                   exclude_status: Optional[Union[str, List[str]]] = None,
                                   prefix: str = None):
        """Generates SQL condition expressions and parameter values for filtering documents by status.

Builds WHERE clause components using the given inclusion and exclusion statuses. Supports field name prefixing for use in joined queries.

Args:
    status (str or list of str): Document status(es) to include. If set to "all", no inclusion condition will be applied.
    exclude_status (str or list of str, optional): Status(es) to exclude. Must not be "all".
    prefix (str, optional): Optional field prefix (e.g., table alias) to prepend to the status field.

**Returns:**

- Tuple[str, list]: A tuple containing the SQL condition string and its corresponding parameter values.
"""
        conds, params = [], []
        prefix = f'{prefix}.' if prefix else ''
        if isinstance(status, str):
            if status != DocListManager.Status.all:
                conds.append(f'{prefix}status = ?')
                params.append(status)
        elif isinstance(status, (tuple, list)):
            conds.append(f'{prefix}status IN ({",".join("?" * len(status))})')
            params.extend(status)

        if isinstance(exclude_status, str):
            assert exclude_status != DocListManager.Status.all, 'Invalid status provided'
            conds.append(f'{prefix}status != ?')
            params.append(exclude_status)
        elif isinstance(exclude_status, (tuple, list)):
            conds.append(f'{prefix}status NOT IN ({",".join("?" * len(exclude_status))})')
            params.extend(exclude_status)

        return ' AND '.join(conds), params

    def _get_all_docs(self):
        with self._db_lock, self._Session() as session:
            return session.query(KBDocument).all()

    def _get_docs(self, to_be_added_doc_ids: List, to_be_deleted_doc_ids: List, filter_status_list: List):
        with self._db_lock, self._Session() as session:
            docs_not_expected = session.query(KBDocument).filter(KBDocument.doc_id.in_(to_be_added_doc_ids)).all()
            docs_expected = session.query(KBDocument).filter(KBDocument.doc_id.in_(to_be_deleted_doc_ids),
                                                             KBDocument.status.in_(filter_status_list)).all()
        return docs_not_expected, docs_expected

    def validate_paths(self, paths: List[str]) -> Tuple[bool, str, List[bool]]:
        """Validates whether the documents corresponding to the given paths can be safely added to the database.

The method checks if the document already exists. If it exists, it verifies whether the document is currently
being parsed, waiting to be parsed, or was not successfully re-parsed last time.

Args:
    paths (List[str]): A list of file paths to validate.

**Returns:**

- Tuple[bool, str, List[bool]]: 
    - bool: Whether all paths passed validation.
    - str: Description message of the validation result.
    - List[bool]: A boolean list corresponding to input paths, indicating whether each path is new (True) or already exists (False).
      If validation fails, this value is None.
"""
        # check and return: success, msg, path_is_new for each path
        unsafe_staus_set = set([DocListManager.Status.working, DocListManager.Status.waiting])
        paths_is_new = [True] * len(paths)
        doc_ids = [gen_docid(path) for path in paths]
        doc_id_to_path = {doc_id: path for doc_id, path in zip(doc_ids, paths)}
        found_doc_ids = []
        found_doc_group_rows = []
        with self._db_lock, self._Session() as session:
            rows = session.execute(
                select(KBDocument.doc_id).where(KBDocument.doc_id.in_(doc_ids))
            ).fetchall()
            if len(rows) == 0:
                return True, 'Success', paths_is_new
            found_doc_ids = [row.doc_id for row in rows]
            found_doc_group_rows = session.execute(
                select(KBGroupDocuments.doc_id, KBGroupDocuments.need_reparse, KBGroupDocuments.status)
                .where(KBGroupDocuments.doc_id.in_(found_doc_ids))).fetchall()

        for doc_group_record in found_doc_group_rows:
            if doc_group_record.need_reparse:
                msg = f'Failed: {doc_id_to_path[doc_group_record.doc_id]} lasttime reparsing has not been finished'
                return False, msg, None
            if doc_group_record.status in unsafe_staus_set:
                return False, f'Failed: {doc_id_to_path[doc_group_record.doc_id]} is being parsed by kbgroup', None
        found_doc_ids = set(found_doc_ids)
        for i in range(len(paths)):
            cur_doc_id = doc_ids[i]
            if cur_doc_id in found_doc_ids:
                paths_is_new[i] = False
        return True, 'Success', paths_is_new

    def update_need_reparsing(self, doc_id: str, need_reparse: bool, group_name: Optional[str] = None):
        """Updates the re-parsing flag for a specific document.

This method sets whether a document should be re-parsed. If a group name is provided, the update is scoped to that group only.

Args:
    doc_id (str): The unique identifier of the document.
    need_reparse (bool): Whether the document needs to be re-parsed.
    group_name (Optional[str]): Optional. The knowledge base group name to filter by. If provided, only documents in the specified group will be updated.
"""
        with self._db_lock, self._Session() as session:
            stmt = update(KBGroupDocuments).where(KBGroupDocuments.doc_id == doc_id)
            if group_name is not None: stmt = stmt.where(KBGroupDocuments.group_name == group_name)
            session.execute(stmt.values(need_reparse=need_reparse))
            session.commit()

    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = DocListManager.Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None):
        """Lists files in the document database based on status filters and returns either full records or file paths.

Args:
    limit (Optional[int]): The maximum number of records to return. If None, all matching records are returned.
    details (bool): Whether to return full database rows or just file paths (document IDs).
    status (Union[str, List[str]]): Status values to include in the result. Defaults to including all.
    exclude_status (Optional[Union[str, List[str]]]): Status values to exclude from the result.

**Returns:**

- list: A list of file records or document paths depending on the `details` flag.
"""
        query = 'SELECT * FROM documents'
        params = []
        status_cond, status_params = self.get_status_cond_and_params(status, exclude_status, prefix=None)
        if status_cond:
            query += f' WHERE {status_cond}'
            params.extend(status_params)
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall() if details else [row[0] for row in cursor]

    def get_docs(self, doc_ids: List[str]) -> List[KBDocument]:
        """Fetches document objects from the database corresponding to the given list of document IDs.

Args:
    doc_ids (List[str]): A list of document IDs to query.

**Returns:**

- List[KBDocument]: A list of matching document objects. Returns an empty list if no matches found.
"""
        with self._db_lock, self._Session() as session:
            docs = session.query(KBDocument).filter(KBDocument.doc_id.in_(doc_ids)).all()
            return docs
        return []

    def set_docs_new_meta(self, doc_meta: Dict[str, dict]):
        """Batch updates the metadata (meta) of documents, and simultaneously updates the new_meta field of documents in knowledge base groups for documents that are not in waiting status.

Args:
    doc_meta (Dict[str, dict]): A dictionary mapping document IDs to their new metadata dictionaries.
"""
        data_to_update = [{'_doc_id': k, '_meta': json.dumps(v)} for k, v in doc_meta.items()]
        with self._db_lock, self._Session() as session:
            # Use sqlalchemy core bulk update
            stmt = KBDocument.__table__.update().where(
                KBDocument.doc_id == bindparam('_doc_id')).values(meta=bindparam('_meta'))
            session.execute(stmt, data_to_update)
            session.commit()

            stmt = KBGroupDocuments.__table__.update().where(
                KBGroupDocuments.doc_id == bindparam('_doc_id'),
                KBGroupDocuments.status != DocListManager.Status.waiting).values(new_meta=bindparam('_meta'))
            session.execute(stmt, data_to_update)
            session.commit()

    def fetch_docs_changed_meta(self, group: str) -> List[DocMetaChangedRow]:
        """Fetches the list of documents within a specified knowledge base group that have updated metadata, and resets the new_meta field for those documents.

Args:
    group (str): Name of the knowledge base group.

**Returns:**

- List[DocMetaChangedRow]: A list containing document IDs and their updated metadata.
"""
        rows = []
        conds = [KBGroupDocuments.group_name == group, KBGroupDocuments.new_meta.isnot(None)]
        with self._db_lock, self._Session() as session:
            rows = (
                session.query(KBDocument.doc_id, KBGroupDocuments.new_meta)
                .join(KBGroupDocuments, KBDocument.doc_id == KBGroupDocuments.doc_id)
                .filter(*conds).all()
            )
            stmt = update(KBGroupDocuments).where(sqlalchemy.and_(*conds)).values(new_meta=None)
            session.execute(stmt)
            session.commit()
        return rows

    def list_all_kb_group(self):
        """Lists all knowledge base group names stored in the database.

**Returns:**

- List[str]: A list of knowledge base group names.
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT group_name FROM document_groups')
            return [row[0] for row in cursor]

    def add_kb_group(self, name):
        """Adds a new knowledge base group name to the database; ignores if the group already exists.

Args:
    name (str): The name of the knowledge base group to add.
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute('INSERT OR IGNORE INTO document_groups (group_name) VALUES (?)', (name,))
            conn.commit()

    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = DocListManager.Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: Union[str, List[str]] = DocListManager.Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None,
                            need_reparse: Optional[bool] = None):
        """Lists files in a specified knowledge base group, with support for multiple filters.

Args:
    group (str, optional): Knowledge base group name to filter by. If None, no group filtering is applied.
    limit (int, optional): Limit on the number of files to return.
    details (bool): Whether to return detailed file information.
    status (str or List[str], optional): Filter files by group document status.
    exclude_status (str or List[str], optional): Exclude files with these group document statuses.
    upload_status (str or List[str], optional): Filter files by upload document status.
    exclude_upload_status (str or List[str], optional): Exclude files with these upload document statuses.
    need_reparse (bool, optional): If set, only returns files marked as needing reparse.

**Returns:**

- list: 
    - If details is False, returns a list of tuples (doc_id, path).
    - If details is True, returns a list of tuples containing detailed file information:
      document ID, path, status, metadata, group name, group status, and group log.
"""
        query = '''
            SELECT documents.doc_id, documents.path, documents.status, documents.meta,
                   kb_group_documents.group_name, kb_group_documents.status, kb_group_documents.log
            FROM kb_group_documents
            JOIN documents ON kb_group_documents.doc_id = documents.doc_id
        '''
        conds, params = [], []
        if group:
            conds.append('kb_group_documents.group_name = ?')
            params.append(group)

        if need_reparse is not None:
            conds.append('kb_group_documents.need_reparse = ?')
            params.append(int(need_reparse))

        status_cond, status_params = self.get_status_cond_and_params(status, exclude_status, prefix='kb_group_documents')
        if status_cond:
            conds.append(status_cond)
            params.extend(status_params)

        status_cond, status_params = self.get_status_cond_and_params(
            upload_status, exclude_upload_status, prefix='documents')
        if status_cond:
            conds.append(status_cond)
            params.extend(status_params)

        if conds: query += ' WHERE ' + ' AND '.join(conds)

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        if not details: return [row[:2] for row in rows]
        return rows

    def delete_unreferenced_doc(self):
        """Deletes documents from the database that are marked for deletion and are no longer referenced by any knowledge base group.

This method queries documents with status "deleting" and a reference count of zero, deletes them from the database,
and adds operation logs for these deletions.

"""
        with self._db_lock, self._Session() as session:
            docs_to_delete = (
                session.query(KBDocument)
                .filter(KBDocument.status == DocListManager.Status.deleting, KBDocument.count == 0)
                .all()
            )
            for doc in docs_to_delete:
                session.delete(doc)
                log = KBOperationLogs(log=f'Delete obsolete file, doc_id:{doc.doc_id}, path:{doc.path}.')
                session.add(log)
            session.commit()

    def _add_doc_records(self, files: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                         status: Optional[str] = DocListManager.Status.waiting, batch_size: int = 64):
        documents = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size] if metadatas else [None] * batch_size
            vals = []

            for i, file_path in enumerate(batch_files):
                doc_id = gen_docid(file_path)

                metadata = batch_metadatas[i].copy() if batch_metadatas[i] else {}
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, file_path)

                vals.append(
                    {
                        KBDocument.doc_id.name: doc_id,
                        KBDocument.filename.name: os.path.basename(file_path),
                        KBDocument.path.name: file_path,
                        KBDocument.meta.name: json.dumps(metadata),
                        KBDocument.status.name: status,
                        KBDocument.count.name: 0,
                    }
                )
            with self._db_lock, self._Session() as session:
                rows = session.execute(
                    insert(KBDocument)
                    .values(vals)
                    .prefix_with('OR IGNORE')
                    .returning(KBDocument.doc_id, KBDocument.path)
                ).fetchall()
                session.commit()
                documents.extend(rows)
        return documents

    def get_docs_need_reparse(self, group: str) -> List[KBDocument]:
        """Retrieves the list of documents that require re-parsing within a specified knowledge base group.

Only documents with status "success" or "failed" and marked as needing reparse in the group are returned.

Args:
    group (str): Name of the knowledge base group.

**Returns:**

- List[KBDocument]: List of documents that need to be re-parsed.
"""
        with self._db_lock, self._Session() as session:
            filter_status_list = [DocListManager.Status.success, DocListManager.Status.failed]
            documents = (
                session.query(KBDocument).join(KBGroupDocuments, KBDocument.doc_id == KBGroupDocuments.doc_id)
                .filter(KBGroupDocuments.need_reparse.is_(True),
                        KBGroupDocuments.group_name == group,
                        KBGroupDocuments.status.in_(filter_status_list)).all())
            return documents
        return []

    def get_existing_paths_by_pattern(self, pattern: str) -> List[str]:
        """Retrieves a list of existing document paths that match a given pattern.

Args:
    pattern (str): Path matching pattern, supports SQL LIKE wildcards.

**Returns:**

- List[str]: List of existing document paths matching the pattern.
"""
        exist_paths = []
        with self._db_lock, self._Session() as session:
            docs = session.query(KBDocument).filter(KBDocument.path.like(pattern)).all()
            exist_paths = [doc.path for doc in docs]
        return exist_paths

    # TODO(wangzhihong): set to metadatas and enable this function
    def update_file_message(self, fileid: str, **kw):
        """Updates fields of the specified file record.

Args:
    fileid (str): Unique identifier of the file (doc_id).
    **kw: Key-value pairs of fields to update and their new values.
"""
        set_clause = ', '.join([f'{k} = ?' for k in kw.keys()])
        params = list(kw.values()) + [fileid]
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute(f'UPDATE documents SET {set_clause} WHERE doc_id = ?', params)
            conn.commit()

    def update_file_status(self, file_ids: List[str], status: str,
                           cond_status_list: Union[None, List[str]] = None) -> List[DocPartRow]:
        """Updates the status of multiple files, optionally filtered by current status.

Args:
    file_ids (List[str]): List of file IDs to update.
    status (str): New status to set.
    cond_status_list (Union[None, List[str]], optional): List of statuses to filter files that can be updated. Defaults to None.

**Returns:**

- List[DocPartRow]: List of updated file IDs and their paths.
"""
        rows = []
        if cond_status_list is None:
            sql_cond = KBDocument.doc_id.in_(file_ids)
        else:
            sql_cond = sqlalchemy.and_(KBDocument.status.in_(cond_status_list), KBDocument.doc_id.in_(file_ids))
        with self._db_lock, self._Session() as session:
            stmt = (
                update(KBDocument)
                .where(sql_cond)
                .values(status=status)
                .returning(KBDocument.doc_id, KBDocument.path)
            )
            rows = session.execute(stmt).fetchall()
            session.commit()
        return rows

    def add_files_to_kb_group(self, file_ids: List[str], group: str):
        """Adds multiple files to the specified knowledge base group.

This method sets the file status to waiting.
If successfully added, increments the document's count.

Args:
    file_ids (List[str]): List of file IDs to add.
    group (str): Name of the knowledge base group.
"""
        with self._db_lock, self._Session() as session:
            vals = []
            for doc_id in file_ids:
                vals = {
                    KBGroupDocuments.doc_id.name: doc_id,
                    KBGroupDocuments.group_name.name: group,
                    KBGroupDocuments.status.name: DocListManager.Status.waiting,
                }
                rows = session.execute(
                    insert(KBGroupDocuments).values(vals).prefix_with('OR IGNORE').returning(KBGroupDocuments.doc_id)
                ).fetchall()
                session.commit()
                if not rows:
                    continue
                doc = session.query(KBDocument).filter_by(doc_id=rows[0].doc_id).one()
                doc.count += 1
                session.commit()

    def delete_files_from_kb_group(self, file_ids: List[str], group: str):
        """Deletes multiple files from the specified knowledge base group.

After deletion, decrements the document's count but not below zero.
If the document is not found, logs a warning.

Args:
    file_ids (List[str]): List of file IDs to delete.
    group (str): Name of the knowledge base group.
"""
        with self._db_lock, self._Session() as session:
            for doc_id in file_ids:
                records_to_delete = (
                    session.query(KBGroupDocuments)
                    .filter(KBGroupDocuments.doc_id == doc_id, KBGroupDocuments.group_name == group)
                    .all()
                )
                for record in records_to_delete:
                    session.delete(record)
                session.commit()
                if not records_to_delete:
                    continue
                try:
                    doc = session.query(KBDocument).filter_by(doc_id=records_to_delete[0].doc_id).one()
                    doc.count = max(0, doc.count - 1)
                    session.commit()
                except NoResultFound:
                    lazyllm.LOG.warning(f'No document found for {doc_id}')

    def get_file_status(self, fileid: str):
        """Gets the status of a specified file.

Args:
    fileid (str): Unique identifier of the file.

**Returns:**

- Optional[Tuple]: A tuple containing the status, or None if the file does not exist.
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.execute('SELECT status FROM documents WHERE doc_id = ?', (fileid,))
        return cursor.fetchone()

    def update_kb_group(self, cond_file_ids: List[str], cond_group: Optional[str] = None,
                        cond_status_list: Optional[List[str]] = None, new_status: Optional[str] = None,
                        new_need_reparse: Optional[bool] = None) -> List[GroupDocPartRow]:
        """Updates the status and reparse need flag of specified files in a knowledge base group.

Batch updates files' status and need_reparse flag within a knowledge base group based on file IDs, group name, and optional status filter.

Args:
    cond_file_ids (List[str]): List of file IDs to update.
    cond_group (Optional[str]): Group name to filter files, if specified only updates files in this group.
    cond_status_list (Optional[List[str]]): Only update files whose status is in this list.
    new_status (Optional[str]): New status to set.
    new_need_reparse (Optional[bool]): New flag indicating if reparse is needed.

**Returns:**

- List[Tuple]: List of tuples of updated files containing doc_id, group_name, and status.
"""
        rows = []
        conds = []
        if not cond_file_ids:
            return rows
        conds.append(KBGroupDocuments.doc_id.in_(cond_file_ids))
        if cond_group is not None:
            conds.append(KBGroupDocuments.group_name == cond_group)
        if cond_status_list:
            conds.append(KBGroupDocuments.status.in_(cond_status_list))

        vals = {}
        if new_status is not None:
            vals[KBGroupDocuments.status.name] = new_status
        if new_need_reparse is not None:
            vals[KBGroupDocuments.need_reparse.name] = new_need_reparse

        if not vals:
            return rows
        with self._db_lock, self._Session() as session:
            stmt = (
                update(KBGroupDocuments)
                .where(sqlalchemy.and_(*conds))
                .values(vals)
                .returning(KBGroupDocuments.doc_id, KBGroupDocuments.group_name, KBGroupDocuments.status)
            )
            rows = session.execute(stmt).fetchall()
            session.commit()
        return rows

    def release(self):
        """Clears all documents, groups, and operation logs from the database.

This operation deletes all records from documents, document_groups, kb_group_documents, and operation_logs tables.
"""
        with self._db_lock, sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread) as conn:
            conn.execute('delete from documents')
            conn.execute('delete from document_groups')
            conn.execute('delete from kb_group_documents')
            conn.execute('delete from operation_logs')
            conn.commit()

    def __reduce__(self):
        return (__class__, (self._path, self._name, self._enable_path_monitoring))


DocListManager.__pool__ = dict(sqlite=SqliteDocListManager)


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description='API status code')
    msg: str = pydantic.Field('success', description='API status message')
    data: Any = pydantic.Field(None, description='API data')

    class Config:
        json_schema_extra = {
            'example': {
                'code': 200,
                'msg': 'success',
            }
        }


def run_in_thread_pool(func: Callable, params: Optional[List[Dict]] = None) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params or []:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()


Default_Suport_File_Types = ['.docx', '.pdf', '.txt', '.json']


def _save_file(_file: 'fastapi.UploadFile', _file_path: str):
    file_content = _file.file.read()
    with open(_file_path, 'wb') as f:
        f.write(file_content)


def _convert_path_to_underscores(file_path: str) -> str:
    return file_path.replace('/', '_').replace('\\', '_')


def _save_file_to_cache(
    file: 'fastapi.UploadFile', cache_dir: str, suport_file_types: List[str]
) -> list:
    to_file_path = os.path.join(cache_dir, file.filename)

    sub_result_list_real_name = []
    if file.filename.endswith('.tar'):

        def unpack_archive(tar_file_path: str, extract_folder_path: str):

            out_file_names = []
            try:
                with tarfile.open(tar_file_path, 'r') as tar:
                    file_info_list = tar.getmembers()
                    for file_info in list(file_info_list):
                        file_extension = os.path.splitext(file_info.name)[-1]
                        if file_extension in suport_file_types:
                            tar.extract(file_info.name, path=extract_folder_path)
                            out_file_names.append(file_info.name)
            except tarfile.TarError as e:
                lazyllm.LOG.error(f'untar error: {e}')
                raise e

            return out_file_names

        _save_file(file, to_file_path)
        out_file_names = unpack_archive(to_file_path, cache_dir)
        sub_result_list_real_name.extend(out_file_names)
        os.remove(to_file_path)
    else:
        file_extension = os.path.splitext(file.filename)[-1]
        if file_extension in suport_file_types:
            if not os.path.exists(to_file_path):
                _save_file(file, to_file_path)
            sub_result_list_real_name.append(file.filename)
    return sub_result_list_real_name


def save_files_in_threads(
    files: List['fastapi.UploadFile'],
    override: bool,
    source_path,
    suport_file_types: List[str] = Default_Suport_File_Types,
):
    real_dir = source_path
    cache_dir = os.path.join(source_path, 'cache')

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    for dir in [real_dir, cache_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    param_list = [
        {'file': file, 'cache_dir': cache_dir, 'suport_file_types': suport_file_types}
        for file in files
    ]

    result_list = []
    for result in run_in_thread_pool(_save_file_to_cache, params=param_list):
        result_list.extend(result)

    already_exist_files = []
    new_add_files = []
    overwritten_files = []

    for file_name in result_list:
        real_file_path = os.path.join(real_dir, _convert_path_to_underscores(file_name))
        cache_file_path = os.path.join(cache_dir, file_name)

        if os.path.exists(real_file_path):
            if not override:
                already_exist_files.append(file_name)
            else:
                os.rename(cache_file_path, real_file_path)
                overwritten_files.append(file_name)
        else:
            os.rename(cache_file_path, real_file_path)
            new_add_files.append(file_name)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return (already_exist_files, new_add_files, overwritten_files)

# returns a list of modified nodes
def parallel_do_embedding(embed: Dict[str, Callable], embed_keys: Optional[Union[List[str], Set[str]]],
                          nodes: List[DocNode], group_embed_keys: Dict[str, List[str]] = None) -> List[DocNode]:
    modified_nodes = []
    with ThreadPoolExecutor(config['max_embedding_workers']) as executor:
        futures = []
        for node in nodes:
            if group_embed_keys:
                embed_keys = group_embed_keys.get(node._group)
                if not embed_keys:
                    continue
            miss_keys = node.has_missing_embedding(embed_keys)
            if not miss_keys:
                continue
            modified_nodes.append(node)
            for k in miss_keys:
                with node._lock:
                    if node.has_missing_embedding(k):
                        future = executor.submit(node.do_embedding, {k: embed[k]}) \
                            if k not in node._embedding_state else executor.submit(node.check_embedding_state, k)
                        node._embedding_state.add(k)
                        futures.append(future)
        if len(futures) > 0:
            for future in concurrent.futures.as_completed(futures):
                future.result()
    return modified_nodes

class _FileNodeIndex(IndexBase):
    def __init__(self):
        self._file_node_map = {}  # Dict[path, Dict[uid, DocNode]]

    @override
    def update(self, nodes: List[DocNode]) -> None:
        for node in nodes:
            path = node.global_metadata.get(RAG_DOC_PATH)
            if path:
                self._file_node_map.setdefault(path, {}).setdefault(node._uid, node)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        for path in list(self._file_node_map.keys()):
            uid2node = self._file_node_map[path]
            for uid in uids:
                uid2node.pop(uid, None)
            if not uid2node:
                del self._file_node_map[path]

    @override
    def query(self, files: List[str]) -> List[DocNode]:
        ret = []
        for file in files:
            nodes = self._file_node_map.get(file)
            if nodes:
                ret.extend(list(nodes.values()))
        return ret

def generic_process_filters(nodes: List[DocNode], filters: Dict[str, Union[str, int, List, Set]]) -> List[DocNode]:
    res = []
    for node in nodes:
        for name, candidates in filters.items():
            value = node.global_metadata.get(name)
            if (not isinstance(candidates, list)) and (not isinstance(candidates, set)):
                if value != candidates:
                    break
            elif (not value) or (value not in candidates):
                break
        else:
            res.append(node)
    return res

def sparse2normal(embedding: Union[Dict[int, float], List[Tuple[int, float]]], dim: int) -> List[float]:
    if not embedding:
        return []

    new_embedding = [0] * dim
    if isinstance(embedding, dict):
        for idx, val in embedding.items():
            new_embedding[int(idx)] = val
    elif isinstance(embedding, list) and isinstance(embedding[0], tuple):
        for pair in embedding:
            new_embedding[int(pair[0])] = pair[1]
    else:
        raise TypeError(f'unsupported embedding datatype `{type(embedding[0])}`')

    return new_embedding

def is_sparse(embedding: Union[Dict[int, float], List[Tuple[int, float]], List[float]]) -> bool:
    if isinstance(embedding, dict):
        return True

    if not isinstance(embedding, list):
        raise TypeError(f'unsupported embedding type `{type(embedding)}`')

    if len(embedding) == 0:
        raise ValueError('empty embedding type is not determined.')

    if isinstance(embedding[0], tuple):
        return True

    if isinstance(embedding[0], list):
        return False

    if isinstance(embedding[0], float) or isinstance(embedding[0], int):
        return False

    raise TypeError(f'unsupported embedding type `{type(embedding[0])}`')


def ensure_call_endpoint(raw: str, *, default_path: str = '/_call') -> str:
    if not raw:
        return raw

    raw = raw.strip()
    has_scheme = '://' in raw
    parts = urlsplit(raw if has_scheme else f'//{raw}', allow_fragments=True)

    if not parts.netloc:
        raise ValueError(f'Invalid endpoint (missing host): {raw}')

    scheme = parts.scheme or 'http'
    new_path = default_path
    return urlunsplit((scheme, parts.netloc, new_path, parts.query, parts.fragment))
