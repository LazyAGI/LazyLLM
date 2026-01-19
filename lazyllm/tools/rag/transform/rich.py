from .base import NodeTransform
from ..doc_node import RichDocNode, DocNode
from typing import List


class RichTransform(NodeTransform):
    def transform(self, node: RichDocNode, **kwargs) -> List[DocNode]:
        assert isinstance(node, RichDocNode), f'Expected RichDocNode, got {type(node)}'
        splitted_nodes = []
        for sub_node in node.nodes:
            new_node = DocNode(content=sub_node.text, metadata=sub_node.metadata, global_metadata=sub_node.global_metadata)
            new_node.excluded_embed_metadata_keys = sub_node.excluded_embed_metadata_keys
            new_node.excluded_llm_metadata_keys = sub_node.excluded_llm_metadata_keys
            splitted_nodes.append(new_node)
        return splitted_nodes
