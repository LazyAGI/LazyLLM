from typing import Dict, Any, List
import requests
from ..module import ModuleBase

class OnlineEmbeddingModuleBase(ModuleBase):

    def __init__(self,
                 embed_url: str,
                 api_key: str,
                 embed_model_name: str,
                 return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._embed_url = embed_url
        self._api_key = api_key
        self._embed_model_name = embed_model_name
        self._set_headers()

    def _set_headers(self) -> Dict[str, str]:
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

    def forward(self, text: str, **kwargs) -> List[float]:
        data = self._encapsulated_data(text, **kwargs)
        with requests.post(self._embed_url, json=data, headers=self._headers) as r:
            if r.status_code == 200:
                return self._parse_response(r.json())
            else:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

    def _encapsulated_data(self, text: str, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": text,
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['data'][0]['embedding']
