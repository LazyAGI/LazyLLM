from typing import Dict, Any, List, Union
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
                 instruct: str = None,
                 instruct_format: str = 'Instruct: {instruct}\nQuery: {text}'
                 ):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._embed_url = embed_url
        self._api_key = api_key
        self._embed_model_name = embed_model_name
        self._instruct = instruct
        self._instruct_format = instruct_format
        self._set_headers()

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

    def _set_instruct_for_input(self, input: Union[List, str]) -> Union[List, str]:
        if isinstance(input, str):
            return self._get_detailed_instruction(input)
        elif isinstance(input, list):
            return [self._get_detailed_instruction(i) for i in input]
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

    def _get_detailed_instruction(self, text: str) -> str:
        return self._instruct_format.format(instruct=self._instruct, text=text)

    def forward(self, input: Union[List, str], **kwargs) -> List[float]:
        is_query = kwargs.pop('is_query', False)
        if is_query and self._instruct:
            input = self._set_instruct_for_input(input)

        data = self._encapsulated_data(input, **kwargs)
        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        with requests.post(self._embed_url, json=data, headers=self._headers, proxies=proxies) as r:
            if r.status_code == 200:
                return self._parse_response(r.json())
            else:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        json_data = {
            "input": input,
            "model": self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        return response['data'][0]['embedding']
