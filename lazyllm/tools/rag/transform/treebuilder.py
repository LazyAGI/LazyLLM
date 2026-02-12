import re
from typing import Any, List, Optional, Callable

from .base import NodeTransform, RuleSet
from ..doc_node import DocNode

def _default_get_level(node: DocNode) -> int:
    try:
        return int(node.metadata.get('text_level') or 0)
    except (ValueError, TypeError):
        return 0


def _extract_numbering_strict_origin(text: str) -> Optional[List[int]]:
    if not text:
        return None
    first_line = text.split('\n')[0].strip()
    if re.match(r'^\s*[\(（]\s*\d{1,3}\s*[\)）]', first_line):
        return None
    if re.match(r'^\s*\d{1,3}\s*[\)）]', first_line):
        return None

    match = re.match(r'^\s*(\d{1,3})(?:\.(\d{1,3}))*', first_line)
    if not match:
        return None
    num_str = match.group(0).strip()
    if not re.match(r'^\d{1,3}(?:\.\d{1,3})*$', num_str):
        return None
    try:
        return [int(n) for n in num_str.split('.')]
    except ValueError:
        return None


def _default_is_valid_child(parent: DocNode, child: DocNode) -> bool:
    p_nums = _extract_numbering_strict_origin(parent.text)
    c_nums = _extract_numbering_strict_origin(child.text)

    if not p_nums or not c_nums:
        return False
    if len(c_nums) != len(p_nums) + 1:
        return False
    return c_nums[:len(p_nums)] == p_nums


class TreeBuilderParser(NodeTransform):
    def __init__(self, rules: Optional[RuleSet] = None, *, get_level: Optional[Callable[[DocNode], int]] = None,
                 is_valid_child: Optional[Callable[[DocNode, DocNode], bool]] = None, return_trace: bool = False,
                 **kwargs
                 ):
        super().__init__(rules=rules or RuleSet(), return_trace=return_trace, **kwargs)
        self._get_level = get_level or _default_get_level
        self._is_valid_child = is_valid_child or _default_is_valid_child

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        return self._parse_nodes(document, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return 'TreeBuilderParser'

    def _parse_nodes(self, nodes: List[DocNode], **kwargs: Any) -> List[DocNode]:  # noqa: C901
        if not nodes:
            return []

        for node in nodes:
            if 'children' in node.metadata:
                node.metadata.pop('children')

        root = DocNode(text='root', metadata={'text_level': 0})
        stack = [root]

        for node in nodes:
            level = self._get_level(node)

            if level == 0:
                if 'children' not in stack[-1].metadata:
                    stack[-1].metadata['children'] = []
                stack[-1].metadata['children'].append(node)
                continue

            target_parent_index = -1

            for i in range(len(stack) - 1, -1, -1):
                parent = stack[i]
                parent_level = self._get_level(parent)

                if parent_level < level:
                    if self._is_valid_child(parent, node):
                        target_parent_index = i
                        break

            if target_parent_index != -1:
                while len(stack) - 1 > target_parent_index:
                    stack.pop()

                if 'children' not in stack[-1].metadata:
                    stack[-1].metadata['children'] = []
                stack[-1].metadata['children'].append(node)
                stack.append(node)
            else:
                while len(stack) > 1:
                    stack.pop()

                if 'children' not in stack[0].metadata:
                    stack[0].metadata['children'] = []
                stack[0].metadata['children'].append(node)
                stack.append(node)

        return root.metadata.get('children', [])
