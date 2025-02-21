from typing import Dict
import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings",
                 embed_model_name: str = "doubao-embedding-text-240715",
                 api_key: str = None):
        super().__init__("DOUBAO", embed_url, api_key or lazyllm.config["doubao_api_key"], embed_model_name)

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": [text],
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data
