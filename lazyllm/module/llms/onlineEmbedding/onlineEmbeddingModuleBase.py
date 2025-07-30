from typing import Dict, Any, List, Union
import requests
from ...module import ModuleBase

class OnlineEmbeddingModuleBase(ModuleBase):
    """
OnlineEmbeddingModuleBase is the base class for managing embedding model interfaces on open platforms, used for requesting text to obtain embedding vectors. It is not recommended to directly instantiate this class. Specific platform classes should inherit from this class for instantiation.
If you need to support the capabilities of embedding models on a new open platform, please extend your custom class from OnlineEmbeddingModuleBase:

1. If the request and response data formats of the new platform's embedding model are the same as OpenAI's, no additional processing is needed; simply pass the URL and model.

2. If the request or response data formats of the new platform's embedding model differ from OpenAI's, you need to override the _encapsulated_data or _parse_response methods.

3. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .


Examples:
    >>> import lazyllm
    >>> from lazyllm.module import OnlineEmbeddingModuleBase
    >>> class NewPlatformEmbeddingModule(OnlineEmbeddingModuleBase):
    ...     def __init__(self,
    ...                 embed_url: str = '<new platform embedding url>',
    ...                 embed_model_name: str = '<new platform embedding model name>'):
    ...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
    ...
    >>> class NewPlatformEmbeddingModule1(OnlineEmbeddingModuleBase):
    ...     def __init__(self,
    ...                 embed_url: str = '<new platform embedding url>',
    ...                 embed_model_name: str = '<new platform embedding model name>'):
    ...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
    ...
    ...     def _encapsulated_data(self, text:str, **kwargs):
    ...         pass
    ...         return json_data
    ...
    ...     def _parse_response(self, response: dict[str, any]):
    ...         pass
    ...         return embedding
    """
    NO_PROXY = True

    def __init__(self,
                 model_series: str,
                 embed_url: str,
                 api_key: str,
                 embed_model_name: str,
                 return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._embed_url = embed_url
        self._api_key = api_key
        self._embed_model_name = embed_model_name
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

    def forward(self, input: Union[List, str], **kwargs) -> List[float]:
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
