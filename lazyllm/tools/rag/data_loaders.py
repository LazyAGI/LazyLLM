from typing import List, Optional
from .store import DocNode, LAZY_ROOT_NAME
from lazyllm import LOG


class DirectoryReader:
    def __init__(self, input_files: List[str]) -> None:
        self._input_files = input_files

    def load_data(self, input_files: Optional[List[str]] = None) -> List[DocNode]:
        input_files = input_files or self._input_files
        from llama_index.core import SimpleDirectoryReader

        LOG.info(f"DirectoryReader loads data, input files: {input_files}")
        reader = SimpleDirectoryReader(input_files=input_files)
        nodes: List[DocNode] = []
        for doc in reader.load_data():
            node = DocNode(
                text=doc.text,
                group=LAZY_ROOT_NAME,
            )
            node.metadata = doc.metadata
            node.excluded_embed_metadata_keys = doc.excluded_embed_metadata_keys
            node.excluded_llm_metadata_keys = doc.excluded_llm_metadata_keys
            nodes.append(node)
        if not nodes:
            LOG.warning(
                f"No nodes load from path {self.input_files}, please check your data path."
            )
        return nodes
