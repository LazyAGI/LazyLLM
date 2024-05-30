import json
from typing import Any, List

import lazyllm
from llama_index.core.embeddings import BaseEmbedding
from lazyllm import LOG

class LLamaIndexEmbeddingWrapper(BaseEmbedding):
    model: Any

    def __init__(self, module: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = module

    def _query_embedding(self, query):
        embeddings = self.model(query)
        if isinstance(embeddings, lazyllm.LazyLlmResponse):
            embeddings = embeddings.messages

        if isinstance(embeddings, str):
            embeddings = json.loads(embeddings)

        LOG.debug(f"_query_embedding, len:{len(query) if isinstance(query, list) else 1}")
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._query_embedding(query=query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._query_embedding(query=text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._query_embedding(query=texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
