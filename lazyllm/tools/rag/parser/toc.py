from typing import List, Union
from .base import BaseParser


class TocParser(BaseParser):
    '''抽取目录'''
    def __init__(self, target_level: Union[int, List[int]] = None, target: str = 'text_level',
                 return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)
        if target_level:
            self.target_level = target_level if isinstance(target_level, list) else [target_level]
        self.target_level = None
        self.target = target

    @classmethod
    def class_name(cls) -> str:
        return 'TocParser'

    def forward(self, nodes, target_level=None, target=None, **kwargs):
        if target_level is None:
            target_level = kwargs.get('target_level', self.target_level)
        if target is None:
            target = kwargs.get('target', self.target)
        return self._parse_nodes(nodes, target_level, target)

    def _parse_nodes(self, nodes, target_level, target):
        if target_level:
            target_level = target_level if isinstance(target_level, list) else [target_level]

        result = []
        for node in nodes:
            text_level = node.metadata.get(target, 0)
            if target_level is None:
                if text_level > 0:
                    result.append(node)
            else:
                if text_level in target_level:
                    result.append(node)

        for index, node in enumerate(result):
            node.metadata['index'] = index

        return result
