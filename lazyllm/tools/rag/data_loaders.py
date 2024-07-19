from typing import List
from .store import DocNode, LAZY_ROOT_NAME
from lazyllm import LOG


class DirectoryReader:
    def __init__(self, input_files: List[str]):
        self.input_files = input_files

    def load_data(self, group: str = LAZY_ROOT_NAME) -> List["DocNode"]:
        from llama_index.core import SimpleDirectoryReader

        llama_index_docs = SimpleDirectoryReader(
            input_files=self.input_files
        ).load_data()
        nodes: List[DocNode] = []
        for doc in llama_index_docs:
            node = DocNode(
                text=doc.text,
                group=group,
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
