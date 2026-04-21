import hashlib
import inspect
import json
from typing import List, Optional, Dict, Union
from lazyllm import LOG
from lazyllm.common.common import once_wrapper

from .doc_node import DocNode, ImageDocNode
from .store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .dataReader import SimpleDirectoryReader
from collections import defaultdict

type_mapping = {
    DocNode: LAZY_ROOT_NAME,
    ImageDocNode: LAZY_IMAGE_GROUP,
}

class DirectoryReader:
    '''Read local files with the configured reader registry and return document nodes.'''

    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        '''Initialize a directory-backed document reader with local and global reader registries.'''
        self._input_files = input_files
        self._local_readers, self._global_readers = local_readers, global_readers

    def _reader_entry_sig(self, reader) -> str:
        if reader is None:
            return '__none__'
        if inspect.isclass(reader):
            return f'{reader.__module__}.{reader.__qualname__}'
        qualname = getattr(reader, '__qualname__', None)
        module = getattr(reader, '__module__', None)
        if qualname and '<lambda>' not in qualname and '<locals>' not in qualname:
            return f'{module}.{qualname}' if module else qualname
        try:
            return '__lambda__::' + inspect.getsource(reader).strip()
        except (OSError, TypeError):
            return repr(reader)

    def signature(self) -> str:
        local_sig = {k: self._reader_entry_sig(v) for k, v in (self._local_readers or {}).items()}
        global_sig = {k: self._reader_entry_sig(v) for k, v in (self._global_readers or {}).items()}
        payload = json.dumps({'local_readers': local_sig, 'global_readers': global_sig}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @once_wrapper
    def _lazy_init(self):
        self._reader = SimpleDirectoryReader(file_extractor={**self._global_readers, **self._local_readers})

    def load_data(self, input_files: Optional[List[str]] = None, metadatas: Optional[Dict] = None,
                  *, split_nodes_by_type: bool = False) -> List[DocNode]:
        '''Load documents from files and optionally split the result by node type.'''
        self._lazy_init()
        input_files = input_files or self._input_files
        nodes: Union[List[DocNode], Dict[str, List[DocNode]]] = defaultdict(list) if split_nodes_by_type else []
        for doc in self._reader(input_files=input_files, metadatas=metadatas):
            doc._group = type_mapping.get(type(doc), LAZY_ROOT_NAME)
            nodes[doc._group].append(doc) if split_nodes_by_type else nodes.append(doc)
        if not nodes:
            raise ValueError(f'No nodes load from path {input_files}, please check your data path.')
        LOG.info('DirectoryReader loads data done!')
        return nodes
