from lazyllm import ModuleBase
from .document import Document
from .graph_document import GraphDocument
from typing import Union


class GraphRetriever(ModuleBase):
    def __init__(self, doc: Union[Document, GraphDocument], **kwargs):
        super().__init__()
        assert isinstance(doc, (Document, GraphDocument)), 'doc must be a Document or GraphDocument instance'
        self._graph_document = None
        if isinstance(doc, GraphDocument):
            self._graph_document = doc
        elif isinstance(doc, Document):
            self._graph_document = doc._graph_document()
        else:
            raise ValueError('doc must be a Document or GraphDocument instance')

    def forward(self, query: str) -> str:
        return self._graph_document.query(query)

    def __repr__(self):
        return f'GraphRetriever(graph_document={self._graph_document})'
