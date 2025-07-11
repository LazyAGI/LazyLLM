from typing import Any, Dict, List
import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class QwenEmbedding(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = ("https://dashscope.aliyuncs.com/api/v1/services/"
                                   "embeddings/text-embedding/text-embedding"),
                 embed_model_name: str = "text-embedding-v1",
                 api_key: str = None):
        super().__init__("QWEN", embed_url, api_key or lazyllm.config['qwen_api_key'], embed_model_name)

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": {
                "texts": [text]
            },
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['output']['embeddings'][0]['embedding']


class QwenReranking(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = ("https://dashscope.aliyuncs.com/api/v1/services/"
                                   "rerank/text-rerank/text-rerank"),
                 embed_model_name: str = "gte-rerank",
                 api_key: str = None, **kwargs):
        super().__init__("QWEN", embed_url, api_key or lazyllm.config['qwen_api_key'], embed_model_name)

    @property
    def type(self):
        return "ONLINE_RERANK"

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "top_n": top_n,
            },
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        results = response['output']['results']
        return [(result["index"], result["relevance_score"]) for result in results]
