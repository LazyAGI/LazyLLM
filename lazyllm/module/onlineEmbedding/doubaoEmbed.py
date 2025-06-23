from typing import Dict, List, Any, Union
import lazyllm
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings",
                 embed_model_name: str = "doubao-embedding-text-240715",
                 api_key: str = None):
        super().__init__("DOUBAO", embed_url, api_key or lazyllm.config["doubao_api_key"], embed_model_name)


class DoubaoMultimodalEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal",
                 embed_model_name: str = "doubao-embedding-vision-241215",
                 api_key: str = None):
        super().__init__("DOUBAO", embed_url, api_key or lazyllm.config["doubao_api_key"], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{"text": input}]
        elif isinstance(input, List):
            # 验证输入格式，最多为1段文本+1张图片
            if len(input) == 0:
                raise ValueError("Input list cannot be empty")
            if len(input) > 2:
                raise ValueError("Input list must contain at most 2 items (1 text and/or 1 image)")
        else:
            raise ValueError("Input must be either a string or a list of dictionaries")

        json_data = {
            "input": input,
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        # 豆包多模态Embedding返回融合的单个embedding
        return response['data']['embedding']
