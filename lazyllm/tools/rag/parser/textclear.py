from ..doc_node import DocNode
from .base import BaseParser
from typing import List, Any

class NodeTextClear(BaseParser):
    '''
    1. 主要清理可能出现的空节点
    2. 将已知的字符进行转义或删除
    '''
    def __init__(self, num_workers: int = 0, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        return self._parse_nodes(document, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "NodeTextClear"

    def _parse_nodes(
        self,
        nodes: List[DocNode],
        **kwargs: Any,
    ) -> List[DocNode]:
        result = []
        for node in nodes:
            if node.text.strip():
                result.append(node)
        return result
