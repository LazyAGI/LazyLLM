from lazyllm import ModuleBase
from typing import List, Any
from ..doc_node import DocNode

class NodeTextClear(ModuleBase):
    def __init__(self, num_workers: int = 0, **kwargs):
        super().__init__(**kwargs)

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        return self._parse_nodes(document, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return 'NodeTextClear'

    def _parse_nodes(self, nodes: List[DocNode], **kwargs: Any) -> List[DocNode]:
        result = []
        for node in nodes:
            if node.text.strip():
                result.append(node)
        return result
