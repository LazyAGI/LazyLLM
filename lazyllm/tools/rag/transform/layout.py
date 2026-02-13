import itertools
from typing import List, Optional, Callable, Any

from .base import NodeTransform, RuleSet, build_rule
from ..doc_node import DocNode

NO_GROUPING = object()

def _default_layout_group_key(node: DocNode) -> str:
    return node.metadata.get('file_name', '')

def _default_layout_sort_key(node: DocNode) -> Any:
    return node.metadata.get('index', 0)

def _default_layout_post_process(nodes: List[DocNode]) -> List[DocNode]:
    for index, node in enumerate(nodes):
        node._metadata['index'] = index
    return nodes

def _get_simple_layout_rules() -> RuleSet:
    return RuleSet([
        build_rule(
            'noop',
            rule=lambda node: True,
            apply=lambda node, rule: node,
        ),
    ])


class LayoutNodeParser(NodeTransform):
    __requires_all_nodes__ = True

    def __init__(self, rules: Optional[RuleSet] = None, *, group_by: Optional[Callable[[DocNode], Any]] = None,
                 post_process: Optional[Callable[[List[DocNode]], List[DocNode]]] = None,
                 sort_by: Optional[Callable[[DocNode], Any]] = None, return_trace: bool = False, **kwargs
    ):
        if rules is None:
            rules = _get_simple_layout_rules()
        super().__init__(rules=rules, return_trace=return_trace, **kwargs)
        if group_by is NO_GROUPING:
            self._no_grouping = True
            self._group_by = None
        else:
            self._no_grouping = False
            self._group_by = group_by if group_by is not None else _default_layout_group_key
        self._sort_by = sort_by if sort_by is not None else _default_layout_sort_key
        self._post_process = post_process if post_process is not None else _default_layout_post_process

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        result_nodes = []
        if not self._no_grouping:
            nodes = sorted(document, key=self._group_by)
            for _key, group in itertools.groupby(nodes, key=self._group_by):
                grouped_nodes = list(group)
                if not grouped_nodes:
                    continue
                grouped_nodes = sorted(grouped_nodes, key=self._sort_by)
                processed = self.process(
                    grouped_nodes,
                    on_match=lambda n, mr, ctx: mr[1],
                    on_miss=lambda n, ctx: n,
                )
                result_nodes.extend(processed)
        else:
            grouped_nodes = sorted(document, key=self._sort_by)
            processed = self.process(
                grouped_nodes,
                on_match=lambda n, mr, ctx: mr[1],
                on_miss=lambda n, ctx: n,
            )
            result_nodes.extend(processed)

        return self._post_process(result_nodes)
