from typing import List, Optional, Dict
from .doc_node import DocNode
from .store_base import LAZY_ROOT_NAME
from lazyllm import LOG
from .dataReader import SimpleDirectoryReader

class DirectoryReader:
    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._local_readers = local_readers
        self._global_readers = global_readers

    def load_data(self, input_files: Optional[List[str]] = None) -> List[DocNode]:
        input_files = input_files or self._input_files
        file_readers = self._local_readers.copy()
        for key, func in self._global_readers.items():
            if key not in file_readers: file_readers[key] = func
        LOG.info(f"DirectoryReader loads data, input files: {input_files}")
        reader = SimpleDirectoryReader(input_files=input_files, file_extractor=file_readers)
        nodes: List[DocNode] = []
        for doc in reader():
            doc.group = LAZY_ROOT_NAME
            nodes.append(doc)
        if not nodes:
            LOG.warning(
                f"No nodes load from path {self.input_files}, please check your data path."
            )
        return nodes
