import json
from typing import Any, List

from llama_index.core.embeddings import BaseEmbedding

class LLamaIndexEmbeddingWrapper(BaseEmbedding):
    model: Any
    def __init__(self, module:Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = module

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = json.loads(self.model(query))
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = json.loads(self.model(text))
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = json.loads(self.model(
            [text for text in texts]
        ))
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)