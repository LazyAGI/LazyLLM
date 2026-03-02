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
    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._local_readers, self._global_readers = local_readers, global_readers

    @once_wrapper
    def _lazy_init(self):
        self._reader = SimpleDirectoryReader(file_extractor={**self._global_readers, **self._local_readers})

    def load_data(self, input_files: Optional[List[str]] = None, metadatas: Optional[Dict] = None,
                  *, split_nodes_by_type: bool = False) -> List[DocNode]:
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
