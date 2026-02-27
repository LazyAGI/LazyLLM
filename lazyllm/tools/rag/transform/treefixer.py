import re
from typing import List, Optional, Tuple, Any

from ..doc_node import DocNode
from .base import NodeTransform, RuleSet, _Context


_NUMBERING_PATTERNS = [
    (r'^([一二三四五六七八九十百千万]+)\s*、', 'chinese_pause'),
    (r'^[（(]\s*([一二三四五六七八九十百千万]+)\s*[）)]', 'chinese_paren'),
    (r'^(\d+)\s*、', 'digit_pause'),
    (r'^[（(]\s*(\d+)\s*[）)]', 'digit_paren'),
    (r'^(\d+)\s*[）)]', 'digit_right_paren'),
    (r'^(\d+(?:\.\d+)+)', 'multilevel'),
    (r'^(\d+)\.(?:\s|$)', 'digit_dot'),
    (r'^([A-Za-z])[\.\、]', 'letter'),
]

_CHINESE_DIGITS = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}
_CHINESE_UNITS = {'十': 10, '百': 100, '千': 1000, '万': 10000}


class TreeFixerParser(NodeTransform):

    def __init__(self, patterns: Optional[List[Tuple[str, str]]] = None, *, skip_level_under: Optional[int] = None,
                 extra_patterns: Optional[List[Tuple[str, str]]] = None, return_trace: bool = False, **kwargs
    ):
        rules = RuleSet()
        super().__init__(rules=rules, return_trace=return_trace, **kwargs)

        base = patterns if patterns is not None else _NUMBERING_PATTERNS
        if extra_patterns:
            base = list(base) + list(extra_patterns)
        self._compiled_patterns = [(re.compile(p), fmt) for p, fmt in base]
        self._skip_level_under = skip_level_under if skip_level_under is not None else 1
        self._reset_state()

    def _reset_state(self) -> None:
        self._result: List[DocNode] = []
        self._stack: List[DocNode] = []
        self._stack_formats: List[Tuple[str, Any]] = []
        self._format_tracker: dict = {}
        self._last_content_parent: Optional[DocNode] = None

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        if not document:
            return []
        flat_nodes = self._flatten_nodes(document)
        if not flat_nodes:
            return []
        self._reset_state()
        self.process(flat_nodes, self._on_match_handler, self._on_miss_handler)
        self._update_text_levels(self._result)
        return self._result

    def _flatten_nodes(self, nodes: List[DocNode]) -> List[DocNode]:
        result = []
        for node in nodes:
            children = node.metadata.pop('children', [])
            node.metadata['children'] = []
            result.append(node)
            if children:
                result.extend(self._flatten_nodes(children))
        return result

    def _extract_numbering(self, node: DocNode) -> Tuple[Optional[str], Optional[Any]]:
        if not node or not node.text:
            return None, None
        text = node.text.strip()
        if not text:
            return None, None
        for pattern, fmt in self._compiled_patterns:
            match = pattern.match(text)
            if match:
                try:
                    value = self._parse_value(match.group(1), fmt)
                    return fmt, value
                except (ValueError, IndexError):
                    continue
        return None, None

    def _parse_value(self, raw: str, fmt: str) -> Any:
        if fmt in ('chinese_pause', 'chinese_paren'):
            return self._parse_chinese_number(raw)
        if fmt == 'multilevel':
            return tuple(int(x) for x in raw.split('.'))
        if fmt == 'letter':
            return ord(raw.upper()) - ord('A') + 1
        return int(raw)

    def _parse_chinese_number(self, s: str) -> int:
        if not s:
            return 0
        if len(s) == 1:
            if s in _CHINESE_DIGITS:
                return _CHINESE_DIGITS[s]
            if s == '十':
                return 10
            return 1
        result = 0
        temp = 0
        for char in s:
            if char in _CHINESE_DIGITS:
                temp = _CHINESE_DIGITS[char]
            elif char in _CHINESE_UNITS:
                unit = _CHINESE_UNITS[char]
                if temp == 0:
                    temp = 1
                result += temp * unit
                temp = 0
        return result + temp

    def _is_consecutive(self, prev_val: Any, curr_val: Any, fmt: str) -> bool:
        if prev_val is None or curr_val is None:
            return False
        try:
            if fmt == 'multilevel':
                return self._is_multilevel_consecutive(prev_val, curr_val)
            return curr_val == prev_val + 1
        except (TypeError, ValueError):
            return False

    def _is_multilevel_consecutive(self, prev: tuple, curr: tuple) -> bool:
        if not (isinstance(prev, tuple) and isinstance(curr, tuple)):
            return False
        if len(prev) == len(curr):
            return prev[:-1] == curr[:-1] and curr[-1] == prev[-1] + 1
        if len(curr) == len(prev) + 1:
            return curr[:-1] == prev and curr[-1] == 1
        if len(curr) < len(prev):
            prefix_len = len(curr) - 1
            return prev[:prefix_len] == curr[:prefix_len] and curr[-1] == prev[prefix_len] + 1
        return False

    def _should_skip_numbering(self, node: DocNode) -> bool:
        if not node or not node.text:
            return False
        if not getattr(node, 'metadata', None) or node.metadata is None:
            return False
        return node.metadata.get('text_level', 0) < self._skip_level_under

    def _update_text_levels(self, nodes: List[DocNode], level: int = 1) -> None:
        for node in nodes:
            orig = node.metadata.get('text_level', 1)
            if orig >= 1:
                node.metadata['text_level'] = level
            children = node.metadata.get('children', [])
            if children:
                self._update_text_levels(children, level + 1)

    def _on_match_handler(self, node: DocNode, matched: tuple, ctx: _Context) -> DocNode:
        return node

    def _on_miss_handler(self, node: DocNode, ctx: _Context) -> DocNode:
        if self._should_skip_numbering(node):
            self._attach_non_title_node(node, self._last_content_parent)
            return node

        fmt, val = self._extract_numbering(node)
        if fmt is None:
            self._attach_to_stack_top(node)
            self._last_content_parent = node
            return node

        self._last_content_parent = None
        if fmt not in self._format_tracker:
            self._handle_new_format(node, fmt, val)
            return node

        last_val = self._format_tracker[fmt]
        if self._is_consecutive(last_val, val, fmt):
            self._handle_consecutive(node, fmt, val)
        else:
            self._handle_non_consecutive(node, fmt, val)
        return node

    def _attach_to_stack_top(self, node: DocNode) -> None:
        if self._stack:
            self._add_child(self._stack[-1], node)
        else:
            self._result.append(node)

    def _handle_new_format(self, node: DocNode, fmt: str, val: Any) -> None:
        if self._stack:
            self._add_child(self._stack[-1], node)
        else:
            self._result.append(node)
        self._stack.append(node)
        self._stack_formats.append((fmt, val))
        self._format_tracker[fmt] = val

    def _handle_consecutive(self, node: DocNode, fmt: str, val: Any) -> None:
        if fmt == 'multilevel' and isinstance(val, tuple) and isinstance(self._format_tracker.get(fmt), tuple):
            prev_val = self._format_tracker[fmt]
            if len(val) == len(prev_val) + 1 and val[:-1] == prev_val:
                self._handle_multilevel_child(node, fmt, val)
                return
        idx = self._find_format_index(fmt)
        if idx > 0:
            parent = self._stack[idx - 1]
            self._add_child(parent, node)
            self._stack[:] = self._stack[:idx] + [node]
            self._stack_formats[:] = self._stack_formats[:idx] + [(fmt, val)]
        elif idx == 0:
            self._result.append(node)
            self._stack[:] = [node]
            self._stack_formats[:] = [(fmt, val)]
        else:
            if self._stack:
                self._add_child(self._stack[-1], node)
                self._stack.append(node)
                self._stack_formats.append((fmt, val))
            else:
                self._result.append(node)
                self._stack.append(node)
                self._stack_formats.append((fmt, val))
        self._format_tracker[fmt] = val

    def _handle_non_consecutive(self, node: DocNode, fmt: str, val: Any) -> None:
        idx = self._find_format_index(fmt)
        if idx >= 0:
            self._stack[:] = self._stack[:idx]
            self._stack_formats[:] = self._stack_formats[:idx]
        if self._stack:
            self._add_child(self._stack[-1], node)
        else:
            self._result.append(node)
        self._stack.append(node)
        self._stack_formats.append((fmt, val))
        self._format_tracker[fmt] = val

    def _handle_multilevel_child(self, node: DocNode, fmt: str, val: Any) -> None:
        idx = self._find_format_index(fmt)
        if idx >= 0:
            parent = self._stack[idx]
            self._add_child(parent, node)
            self._stack[:] = self._stack[:idx + 1] + [node]
            self._stack_formats[:] = self._stack_formats[:idx + 1] + [(fmt, val)]
        else:
            if self._stack:
                self._add_child(self._stack[-1], node)
                self._stack.append(node)
                self._stack_formats.append((fmt, val))
            else:
                self._result.append(node)
                self._stack.append(node)
                self._stack_formats.append((fmt, val))
        self._format_tracker[fmt] = val

    def _find_format_index(self, fmt: str) -> int:
        for i in range(len(self._stack_formats) - 1, -1, -1):
            if self._stack_formats[i][0] == fmt:
                return i
        return -1

    def _attach_non_title_node(
        self,
        node: DocNode,
        content_parent: Optional[DocNode] = None,
    ) -> None:
        parent = content_parent if content_parent is not None else (self._stack[-1] if self._stack else None)
        if parent is not None:
            self._add_child(parent, node)
        else:
            self._result.append(node)

    def _add_child(self, parent: DocNode, child: DocNode) -> None:
        if parent is None or child is None:
            return
        if not getattr(parent, 'metadata', None) or parent.metadata is None:
            parent.metadata = {}
        if 'children' not in parent.metadata:
            parent.metadata['children'] = []
        parent.metadata['children'].append(child)
