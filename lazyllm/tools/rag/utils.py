import os
import shutil
import hashlib
from typing import List, Callable, Generator, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import pydantic
import sqlite3
from pydantic import BaseModel
from fastapi import UploadFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

import lazyllm
from lazyllm import config


config.add("default_dlmanager", str, "sqlite", "DEFAULT_DOCLIST_MANAGER")

class DocListManager(ABC):
    DEDAULT_GROUP_NAME = '__default__'
    __pool__ = dict()

    class Status:
        all = 'all'
        waiting = 'waiting'
        working = 'working'
        success = 'success'
        failed = 'failed'
        deleting = 'deleting'
        deleting = 'deleted'

    def __init__(self, path, name):
        self._path = path
        self._name = name
        self._id = hashlib.sha256(f'{name}@+@{path}'.encode()).hexdigest()
        if not os.path.isabs(path):
            raise ValueError("directory must be an absolute path")

    def __new__(cls, *args, **kw):
        if cls is not DocListManager:
            return super().__new__(cls)
        return __class__.__pool__[config['default_dlmanager']](*args, **kw)

    def init_tables(self) -> 'DocListManager':
        if not self.table_inited():
            # init tables
            self._init_tables()

            # add files to tables
            files_list = []
            for root, _, files in os.walk(self._path):
                files = [os.path.join(root, file_path) for file_path in files]
                files_list.extend(files)
            ids = self.add_files(files_list, status=DocListManager.Status.success)
            self.add_kb_group(DocListManager.DEDAULT_GROUP_NAME)
            self.add_files_to_kb_group(ids, group=DocListManager.DEDAULT_GROUP_NAME)
        return self

    def delete_files(self, file_ids: List[str]):
        self.update_kb_group_file_status(file_ids=file_ids, status=DocListManager.Status.deleting)
        self._delete_files(file_ids)

    @abstractmethod
    def table_inited(self): pass

    @abstractmethod
    def _init_tables(self): pass

    @abstractmethod
    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None): pass

    @abstractmethod
    def list_all_kb_group(self): pass

    @abstractmethod
    def add_kb_group(self, name): pass

    @abstractmethod
    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: str = Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None): pass

    @abstractmethod
    def add_files(self, files: List[str], metadatas: Optional[List] = None,
                  status: Optional[str] = None) -> List[str]: pass

    @abstractmethod
    def update_file_message(self, fileid: str, **kw): pass

    @abstractmethod
    def add_files_to_kb_group(self, file_ids: List[str], group: str): pass

    @abstractmethod
    def _delete_files(self, file_ids: List[str]): pass

    @abstractmethod
    def delete_files_from_kb_group(self, file_ids: List[str], group: str): pass

    @abstractmethod
    def get_file_status(self, fileid: str): pass

    @abstractmethod
    def update_file_status(self, file_ids: List[str], status: str): pass

    @abstractmethod
    def update_kb_group_file_status(self, file_ids: Union[str, List[str]],
                                    status: str, group: Optional[str] = None): pass

    @abstractmethod
    def release(self): pass


class SqliteDocListManager(DocListManager):
    def __init__(self, path, name):
        super().__init__(path, name)
        root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
        os.system(f'mkdir -p {root_dir}')
        self._db_path = os.path.join(root_dir, f'.lazyllm_dlmanager.{self._id}.db')
        self._conns = threading.local()

    @property
    def _conn(self):
        if not hasattr(self._conns, 'impl'): self._conns.impl = sqlite3.connect(self._db_path)
        return self._conns.impl

    def _init_tables(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT,
                    count INTEGER DEFAULT 0
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS document_groups (
                    group_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name TEXT NOT NULL UNIQUE
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS kb_group_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    classification TEXT,
                    status TEXT,
                    log TEXT,
                    UNIQUE (doc_id, group_name),
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
                    FOREIGN KEY(group_name) REFERENCES document_groups(group_name)
                )
            """)

    def table_inited(self):
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        return cursor.fetchone() is not None

    @staticmethod
    def get_status_cond_and_params(status: Union[str, List[str]],
                                   exclude_status: Optional[Union[str, List[str]]] = None,
                                   prefix: str = None):
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
            conds.append('{prefix}status != ?')
            params.append(exclude_status)
        elif isinstance(exclude_status, (tuple, list)):
            conds.append(f'{prefix}status NOT IN ({",".join("?" * len(exclude_status))})')
            params.extend(exclude_status)

        return ' AND '.join(conds), params

    def list_files(self, limit: Optional[int] = None, details: bool = False,
                   status: Union[str, List[str]] = DocListManager.Status.all,
                   exclude_status: Optional[Union[str, List[str]]] = None):
        query = "SELECT * FROM documents"
        params = []
        status_cond, status_params = self.get_status_cond_and_params(status, exclude_status, prefix=None)
        if status_cond:
            query += f' WHERE {status_cond}'
            params.extend(status_params)
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor = self._conn.execute(query, params)
        return cursor.fetchall() if details else [row[0] for row in cursor]

    def list_all_kb_group(self):
        cursor = self._conn.execute("SELECT group_name FROM document_groups")
        return [row[0] for row in cursor]

    def add_kb_group(self, name):
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO document_groups (group_name) VALUES (?)', (name,))

    def list_kb_group_files(self, group: str = None, limit: Optional[int] = None, details: bool = False,
                            status: Union[str, List[str]] = DocListManager.Status.all,
                            exclude_status: Optional[Union[str, List[str]]] = None,
                            upload_status: str = DocListManager.Status.all,
                            exclude_upload_status: Optional[Union[str, List[str]]] = None):
        query = """
            SELECT documents.doc_id, documents.path, documents.status,
                   kb_group_documents.group_name, kb_group_documents.classification,
                   kb_group_documents.status, kb_group_documents.log
            FROM kb_group_documents
            JOIN documents ON kb_group_documents.doc_id = documents.doc_id
        """
        conds, params = [], []
        if group:
            conds.append('kb_group_documents.group_name = ?')
            params.append(group)

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

        with self._conn:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

        if not details: return [row[:2] for row in rows]
        return rows

    def add_files(self, files: List[str], metadatas: Optional[List] = None, status: Optional[str] = None):
        with self._conn:
            ids = []
            for i, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                metadata = json.dumps(metadatas[i]) if metadatas else ''
                doc_id = hashlib.sha256(f'{file_path}'.encode()).hexdigest()
                with self._conn:
                    self._conn.execute("""
                        INSERT OR IGNORE INTO documents (doc_id, filename, path, metadata, status, count)
                        VALUES (?, ?, ?, ?, ?, ?) RETURNING doc_id;
                    """, (doc_id, filename, file_path, metadata, status or DocListManager.Status.waiting, 1))
                ids.append(doc_id)
            return ids

    # TODO(wangzhihong): set to metadatas and enable this function
    def update_file_message(self, fileid: str, **kw):
        set_clause = ", ".join([f"{k} = ?" for k in kw.keys()])
        params = list(kw.values()) + [fileid]
        with self._conn:
            self._conn.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = ?", params)

    def add_files_to_kb_group(self, file_ids: List[str], group: str):
        with self._conn:
            for doc_id in file_ids:
                self._conn.execute("""
                    INSERT OR IGNORE INTO kb_group_documents (doc_id, group_name, status)
                    VALUES (?, ?, ?)
                """, (doc_id, group, DocListManager.Status.waiting))

    def _delete_files(self, file_ids: List[str]):
        with self._conn:
            for doc_id in file_ids:
                self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def delete_files_from_kb_group(self, file_ids: List[str], group: str):
        with self._conn:
            for doc_id in file_ids:
                self._conn.execute("DELETE FROM kb_group_documents WHERE doc_id = ? AND group_name = ?", (doc_id, group))

    def get_file_status(self, fileid: str):
        with self._conn:
            cursor = self._conn.execute("SELECT status FROM documents WHERE doc_id = ?", (fileid,))
        return cursor.fetchone()

    def update_file_status(self, file_ids: List[str], status: str):
        with self._conn:
            for fileid in file_ids:
                self._conn.execute("UPDATE documents SET status = ? WHERE doc_id = ?", (status, fileid))

    def update_kb_group_file_status(self, file_ids: Union[str, List[str]], status: str, group: Optional[str] = None):
        if isinstance(file_ids, str): file_ids = [file_ids]
        query, params = 'UPDATE kb_group_documents SET status = ? WHERE ', [status]
        if group:
            query += 'group_name = ? AND '
            params.append(group)
        query += f'doc_id IN ({",".join("?" * len(file_ids))})'
        with self._conn:
            self._conn.execute(query, (params + file_ids))

    def release(self):
        self._conn.close()
        os.system(f'rm {self._db_path}')

    def __reduce__(self):
        return (__class__, (self._path, self._name))


DocListManager.__pool__ = dict(sqlite=SqliteDocListManager)


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


def run_in_thread_pool(
    func: Callable,
    params: List[Dict] = [],
) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()


Default_Suport_File_Types = [".docx", ".pdf", ".txt", ".json"]


def _save_file(_file: UploadFile, _file_path: str):
    file_content = _file.file.read()
    with open(_file_path, "wb") as f:
        f.write(file_content)


def _convert_path_to_underscores(file_path: str) -> str:
    return file_path.replace("/", "_").replace("\\", "_")


def _save_file_to_cache(
    file: UploadFile, cache_dir: str, suport_file_types: List[str]
) -> list:
    to_file_path = os.path.join(cache_dir, file.filename)

    sub_result_list_real_name = []
    if file.filename.endswith(".tar"):

        def unpack_archive(tar_file_path: str, extract_folder_path: str):
            import tarfile

            out_file_names = []
            try:
                with tarfile.open(tar_file_path, "r") as tar:
                    file_info_list = tar.getmembers()
                    for file_info in list(file_info_list):
                        file_extension = os.path.splitext(file_info.name)[-1]
                        if file_extension in suport_file_types:
                            tar.extract(file_info.name, path=extract_folder_path)
                            out_file_names.append(file_info.name)
            except tarfile.TarError as e:
                lazyllm.LOG.error(f"untar error: {e}")
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
    files: List[UploadFile],
    override: bool,
    source_path,
    suport_file_types: List[str] = Default_Suport_File_Types,
):
    real_dir = source_path
    cache_dir = os.path.join(source_path, "cache")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    for dir in [real_dir, cache_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    param_list = [
        {"file": file, "cache_dir": cache_dir, "suport_file_types": suport_file_types}
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
