import hashlib
import os
import shutil

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (Any, Callable, Dict, Generator, List, Optional, Set, Tuple,
                    Union)
from urllib.parse import urlsplit, urlunsplit
from numbers import Integral

import pydantic
from lazyllm.thirdparty import fastapi
from pydantic import BaseModel

import lazyllm
from lazyllm import config
from lazyllm.common import override
from lazyllm.thirdparty import tarfile

from .doc_node import DocNode, MetadataMode
from .global_metadata import RAG_DOC_PATH
from .index_base import IndexBase

# min(32, (os.cpu_count() or 1) + 4) is the default number of workers for ThreadPoolExecutor
config.add(
    'max_embedding_workers',
    int,
    min(32, (os.cpu_count() or 1) + 4),
    'MAX_EMBEDDING_WORKERS',
    description='The default number of workers for embedding in RAG.')

RAG_DEFAULT_GROUP_NAME = '__default__'

def gen_docid(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()


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
def parallel_do_embedding(embed: Dict[str, Callable], embed_keys: Optional[Union[List[str], Set[str]]],  # noqa: C901
                          nodes: List[DocNode], group_embed_keys: Dict[str, List[str]] = None) -> List[DocNode]:
    if not nodes: return []

    max_workers = config['max_embedding_workers']

    tasks_by_key: Dict[str, List[DocNode]] = defaultdict(list)
    modified_nodes: List[DocNode] = []
    for node in nodes:
        if group_embed_keys and not group_embed_keys.get(node._group):
            continue
        keys_for_node = (group_embed_keys.get(node._group) if group_embed_keys else embed_keys) or embed.keys()
        miss = node.has_missing_embedding(keys_for_node)
        if not miss:
            continue
        modified_nodes.append(node)
        for k in miss:
            tasks_by_key[k].append(node)
            node.embedding_state.add(k)

    if not tasks_by_key:
        return []
    concurrent_workers = min(max_workers, len(tasks_by_key))
    max_workers_per_key = max(1, max_workers // max(1, concurrent_workers))

    def _check_batch(fn):
        batch_size = getattr(fn, 'batch_size', None)
        return isinstance(batch_size, Integral) and batch_size > 1

    def _check_empty_embedding_item(vec, embed_key: str, idx: int) -> None:
        if vec is None:
            raise ValueError(f'[LazyLLM - parallel_do_embedding][{embed_key}] invalid embedding at index {idx}: None')
        if isinstance(vec, (list, dict)) and len(vec) == 0:
            raise ValueError(f'[LazyLLM - parallel_do_embedding][{embed_key}] '
                             f'invalid embedding at index {idx}: empty {type(vec).__name__}')

    def _process_key(k: str, knodes: List[DocNode]):
        try:
            fn = embed[k]
            if _check_batch(fn):
                texts = [n.get_text(MetadataMode.EMBED) for n in knodes]
                vecs = fn(texts)
                if len(vecs) != len(texts):
                    raise ValueError(f'[LazyLLM - parallel_do_embedding][{k}] batch size mismatch: '
                                     f'[text_num:{len(texts)}] vs [vec_num:{len(vecs)}]')
                for idx, (n, v) in enumerate(zip(knodes, vecs)):
                    _check_empty_embedding_item(v, k, idx)
                    n.set_embedding(k, v)
                return

            if max_workers_per_key == 1:
                for n in knodes:
                    n.do_embedding({k: fn})
            else:
                with ThreadPoolExecutor(max_workers=max_workers_per_key) as ex2:
                    futs = [ex2.submit(n.do_embedding, {k: fn}) for n in knodes]
                    for fut in as_completed(futs):
                        fut.result()
        except Exception as e:
            lazyllm.LOG.error(f'[LazyLLM - parallel_do_embedding][{k}] error: {e}')
            for n in knodes:
                if k in n.embedding_state:
                    with n._lock:
                        n.embedding_state.remove(k)
            raise e

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks_by_key))) as ex:
        futs = [ex.submit(_process_key, k, k_nodes) for k, k_nodes in tasks_by_key.items()]
        for fut in as_completed(futs):
            fut.result()
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

def _get_default_db_config(db_name: str):
    '''get default db config'''
    db_name = db_name.split('.')[0]
    root_dir = os.path.expanduser(os.path.join(config['home'], '.dbs'))
    os.makedirs(root_dir, exist_ok=True)
    db_path = os.path.join(root_dir, f'lazyllm_{db_name}.db')
    return {
        'db_type': 'sqlite',
        'user': None,
        'password': None,
        'host': None,
        'port': None,
        'db_name': db_path,
    }

def _orm_to_dict(obj) -> Dict[str, Any]:
    '''convert ORM object to dict'''
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
