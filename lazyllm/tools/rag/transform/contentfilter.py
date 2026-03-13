from typing import List, Optional

from .base import NodeTransform, RuleSet, Rule
from ..doc_node import DocNode, RichDocNode


DEFAULT_NON_EMPTY_RULE = RuleSet([Rule.build(
    'non_empty',
    rule=lambda n: bool(n.text.strip() if getattr(n, 'text', '') else ''),
    apply=lambda n, r: n,
)])


class ContentFiltParser(NodeTransform):
    def __init__(self, rules: Optional[RuleSet] = None, num_workers: int = 0, **kwargs):
        rules = rules if rules is not None else DEFAULT_NON_EMPTY_RULE
        super().__init__(num_workers=num_workers, rules=rules, **kwargs)

    def forward(self, node: DocNode, **kwargs) -> List[DocNode]:
        nodes = node.nodes if isinstance(node, RichDocNode) else [node]
        results = self.process(
            nodes,
            on_match=lambda node, match, ctx: node,
            on_miss=lambda node, ctx: None,
        )
        return [x for x in results if x is not None]
