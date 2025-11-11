from lazyllm import ModuleBase
from lazyllm.tools.rag.global_metadata import RAG_KB_ID
from .document import Document
from .graph_document import GraphDocument
from typing import Union


class GraphRetriever(ModuleBase):
    def __init__(self, doc: Union[Document, GraphDocument], kb_id: str, **kwargs):
        super().__init__()
        assert isinstance(doc, Document), 'doc must be a Document instance'
        self._kb_id = kb_id
        self._graph_document = None
        if isinstance(doc, GraphDocument):
            self._graph_document = doc
        elif isinstance(doc, Document):
            if doc._graph_documents.get(kb_id, None) is None:
                raise ValueError(f'GraphDocument for kb {kb_id} not found in Document')
            self._graph_document = doc._graph_documents[kb_id]()
        else:
            raise ValueError('doc must be a Document or GraphDocument instance')

    def forward(self, query: str, kb_id: RAG_KB_ID) -> str:
        return self._graph_document.query_graphrag_index_for_kb(query)

    def __repr__(self):
        return f'GraphRetriever(graph_document={self._graph_document})'
