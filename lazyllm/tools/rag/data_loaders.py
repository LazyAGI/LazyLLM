from typing import List, Optional, Dict
from .doc_node import DocNode, ImageDocNode
from .store_base import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from lazyllm import LOG
from .dataReader import SimpleDirectoryReader
from .global_metadata import RAG_DOC_PATH

class DirectoryReader:
    def __init__(self, input_files: Optional[List[str]], local_readers: Optional[Dict] = None,
                 global_readers: Optional[Dict] = None) -> None:
        self._input_files = input_files
        self._local_readers = local_readers
        self._global_readers = global_readers

    def load_data(self, input_files: Optional[List[str]] = None, metadatas: Optional[Dict] = None) -> List[DocNode]:
        input_files = input_files or self._input_files
        file_readers = self._local_readers.copy()
        for key, func in self._global_readers.items():
            if key not in file_readers: file_readers[key] = func
        LOG.info(f"DirectoryReader loads data, input files: {input_files}")
        reader = SimpleDirectoryReader(input_files=input_files, file_extractor=file_readers)
        nodes: List[DocNode] = []
        for doc in reader():
            doc._group = LAZY_IMAGE_GROUP if isinstance(doc, ImageDocNode) else LAZY_ROOT_NAME
            nodes.append(doc)
        if not nodes:
            LOG.warning(
                f"No nodes load from path {input_files}, please check your data path."
            )
        if metadatas:
            map_file_meta = {}
            for file_path, metadata in zip(input_files, metadatas):
                if metadata:
                    map_file_meta[file_path] = metadata
            for node in nodes:
                file_path = node.global_metadata[RAG_DOC_PATH]
                if file_path in map_file_meta:
                    node.global_metadata.update(map_file_meta[file_path])
        LOG.info("DirectoryReader loads data done!")
        return nodes
