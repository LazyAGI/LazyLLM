from .base import NodeTransform
from ..doc_node import RichDocNode, DocNode
from typing import List, Union


class RichTransform(NodeTransform):
    def transform(self, node: RichDocNode, **kwargs) -> List[Union[str, DocNode]]:
        assert isinstance(node, RichDocNode), f'Expected RichDocNode, got {type(node)}'
        return node._nodes
