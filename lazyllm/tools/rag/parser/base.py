from lazyllm import ModuleBase
from ..doc_node import DocNode
from typing import List

class BaseParser(ModuleBase):
    def __init__(self, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)

    def _parse_nodes(self, nodes: List[DocNode]) -> List[DocNode]:
        pass

    def forward(self, nodes: List[DocNode], **kwargs) -> List[DocNode]:
        return self._parse_nodes(nodes)
