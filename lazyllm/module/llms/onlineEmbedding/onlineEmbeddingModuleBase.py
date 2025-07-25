from typing import Dict, Any, List, Union
import requests
from ...module import ModuleBase

class OnlineEmbeddingModuleBase(ModuleBase):
    """OnlineEmbeddingModuleBase是管理开放平台的嵌入模型接口的基类，用于请求文本获取嵌入向量。不建议直接对该类进行直接实例化。需要特定平台类继承该类进行实例化。

如果你需要支持新的开放平台的嵌入模型的能力，请让你自定义的类继承自OnlineEmbeddingModuleBase：

1、如果新平台的嵌入模型的请求和返回数据格式都和openai一样，可以不用做任何处理，只传url和模型即可

2、如果新平台的嵌入模型的请求或者返回的数据格式和openai不一样，需要重写_encapsulated_data或_parse_response方法。

3、配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加


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
