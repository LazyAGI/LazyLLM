from lazyllm import ModuleBase
from lazyllm.tools.rag.global_metadata import RAG_KB_ID
from .document import Document


class GraphRetriever(ModuleBase):
    def __init__(self, doc: Document, **kwargs):
        super().__init__()
        assert isinstance(doc, Document), 'doc must be a Document instance'
        self._doc = doc

    def forward(self, query: str, kb_id: RAG_KB_ID) -> str:
        return self._doc.query_graphrag_index_for_kb(kb_id, query)

    def __repr__(self):
        return f'GraphRetriever(doc={self._doc})'
