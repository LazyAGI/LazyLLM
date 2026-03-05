from .base import NodeTransform
from ..doc_node import RichDocNode, DocNode
from typing import List


class RichTransform(NodeTransform):
    __support_rich__ = True

    def _clone_node(self, n: DocNode) -> DocNode:
        new_node = DocNode(content=n.text, metadata=n.metadata,
                           global_metadata=n.global_metadata)
        new_node.excluded_embed_metadata_keys = n.excluded_embed_metadata_keys
        new_node.excluded_llm_metadata_keys = n.excluded_llm_metadata_keys
        return new_node

    def forward(self, node: DocNode, **kwargs) -> List[DocNode]:
        assert isinstance(node, RichDocNode), f'Expected RichDocNode, got {type(node)}'
        return [self._clone_node(sub_node) for sub_node in node.nodes]
