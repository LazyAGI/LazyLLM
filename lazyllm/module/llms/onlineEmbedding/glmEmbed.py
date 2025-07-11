from typing import Any, Dict, List

import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase


class GLMEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings",
                 embed_model_name: str = "embedding-2",
                 api_key: str = None):
        super().__init__("GLM", embed_url, api_key or lazyllm.config["glm_api_key"], embed_model_name)

class GLMReranking(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = "https://open.bigmodel.cn/api/paas/v4/rerank",
                 embed_model_name: str = "rerank",
                 api_key: str = None):
        super().__init__("GLM", embed_url, api_key or lazyllm.config["glm_api_key"], embed_model_name)

    @property
    def type(self):
        return "ONLINE_RERANK"

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
            "return_raw_scores": True
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return [(result["index"], result["relevance_score"]) for result in response['results']]
