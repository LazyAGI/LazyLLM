import time

from typing import Any, Dict, List

import lazyllm
from lazyllm.thirdparty import jwt
from .onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase

class SenseNovaEmbedding(OnlineEmbeddingModuleBase):

    def __init__(self,
                 embed_url: str = "https://api.sensenova.cn/v1/llm/embeddings",
                 embed_model_name: str = "nova-embedding-stable",
                 api_key: str = None,
                 secret_key: str = None):
        if not api_key and not secret_key:
            api_key, secret_key = lazyllm.config['sensenova_api_key'], lazyllm.config['sensenova_secret_key']
        if secret_key.startswith('sk-'): api_key, secret_key = secret_key, None
        if not api_key.startswith('sk-'):
            if ':' in api_key: api_key, secret_key = api_key.split(':', 1)
            assert secret_key, 'secret_key should be provided with sensecore api_key'
            api_key = SenseNovaEmbedding.encode_jwt_token(api_key, secret_key)
        super().__init__("SENSENOVA", embed_url, api_key, embed_model_name)

    @staticmethod
    def encode_jwt_token(ak: str, sk: str) -> str:
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": ak,
            # Fill in the expected effective time, which represents the current time +24 hours
            "exp": int(time.time()) + 86400,
            # Fill in the desired effective time starting point, which represents the current time
            "nbf": int(time.time())
        }
        token = jwt.encode(payload, sk, headers=headers)
        return token

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": [text],
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['embeddings'][0]['embedding']
