from typing import List
from .store import DocNode
from llama_index.core import SimpleDirectoryReader


class DirectoryReader:
    def __init__(self, input_files: List[str]):
        self.input_files = input_files

    def load_data(self) -> List['DocNode']:
        llama_index_docs = SimpleDirectoryReader(input_files=self.input_files).load_data()
        nodes: List[DocNode] = []
        for doc in llama_index_docs:
            node = DocNode(
                text=doc.text,
                metadata=doc.metadata,
                excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys,
            )
            nodes.append(node)
        return nodes
