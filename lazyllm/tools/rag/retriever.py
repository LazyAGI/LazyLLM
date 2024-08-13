from lazyllm import ModuleBase
from .store import DocNode
from typing import List


class Retriever(ModuleBase):
    __enable_request__ = False

    def __init__(
        self,
        doc: object,
        group_name: str,
        similarity: str = "dummy",
        similarity_cut_off: float = float("-inf"),
        index: str = "default",
        topk: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.doc = doc
        self.group_name = group_name
        self.similarity = similarity  # similarity function str
        self.similarity_cut_off = similarity_cut_off
        self.index = index
        self.topk = topk
        self.similarity_kw = kwargs  # kw parameters

    def forward(self, query: str) -> List[DocNode]:
        return self.doc.forward(
            func_name="retrieve",
            query=query,
            group_name=self.group_name,
            similarity=self.similarity,
            similarity_cut_off=self.similarity_cut_off,
            index=self.index,
            topk=self.topk,
            similarity_kws=self.similarity_kw,
        )
