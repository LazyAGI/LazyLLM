from typing import Dict, List, Union
import requests
from ....module import ModuleBase

class OnlineEmbeddingModuleBase(ModuleBase):
    NO_PROXY = True

    def __init__(self,
                 model_series: str,
                 embed_url: str,
                 api_key: str,
                 embed_model_name: str,
                 return_trace: bool = False,
                 **kw):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._embed_url = embed_url
        self._api_key = api_key
        self._embed_model_name = embed_model_name
        self._set_headers()
        self.batch_size = kw.pop('batch_size', 10)

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return "EMBED"

    def _set_headers(self) -> Dict[str, str]:
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

    def forward(self, input: Union[List, str], **kwargs) -> List[float]:
        data = self._encapsulated_data(input, **kwargs)
        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        if isinstance(data, List):
            return self.run_embed_batch(input, data, proxies, **kwargs)
        else:
            with requests.post(self._embed_url, json=data, headers=self._headers, proxies=proxies) as r:
                if r.status_code == 200:
                    return self._parse_response(r.json(), input)
                else:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

    def run_embed_batch(self, input: Union[List, str], data: List, proxies, **kwargs):
        ret = []
        flag = False
        error_msg = ""
        while not flag:
            temp = []
            for i in range(len(data)):
                with requests.post(self._embed_url, json=data[i], headers=self._headers, proxies=proxies) as r:
                    if r.status_code == 200:
                        temp.extend(self._parse_response(r.json(), input))
                        if i == len(data) - 1:
                            flag = True
                    else:
                        flag = False
                        error_msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                        if self.batch_size == 1 or r.status_code == 401:
                            raise requests.RequestException(error_msg)
                        else:
                            self.batch_size = max(self.batch_size // 2, 1)
                            data = self._encapsulated_data(input, **kwargs)
                        break
            ret = temp
        if not flag:
            raise requests.RequestException(error_msg)
        return ret

    def _encapsulated_data(self, input: Union[List, str], **kwargs):
        json_data = {
            "input": input,
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> Union[List[List[float]], List[float]]:
        data = response.get("data", [])
        if not data:
            raise Exception("no data received")
        if isinstance(input, str):
            return data[0].get("embedding", [])
        else:
            return [res.get("embedding", []) for res in data]
