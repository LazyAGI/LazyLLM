from typing import Any, Dict, List
import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class QwenEmbedding(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = "https://dashscope.aliyuncs.com/api/v1/services/\
                                    embeddings/text-embedding/text-embedding",
                 embed_model_name: str = "text-embedding-v1"):
        super().__init__(embed_url, lazyllm.config['qwen_api_key'], embed_model_name)

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
