from typing import Dict, List, Union
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from lazyllm import LOG
from .utils import OnlineModuleBase


class OnlineEmbeddingModuleBase(OnlineModuleBase):
    """
OnlineEmbeddingModuleBase is the base class for managing embedding model interfaces on open platforms, used for requesting text to obtain embedding vectors. It is not recommended to directly instantiate this class. Specific platform classes should inherit from this class for instantiation.


If you need to support the capabilities of embedding models on a new open platform, please extend your custom class from OnlineEmbeddingModuleBase:

    1. If the request and response data formats of the new platform's embedding model are the same as OpenAI's, no additional processing is needed; simply pass the URL and model.
    2. If the request or response data formats of the new platform's embedding model differ from OpenAI's, you need to override the _encapsulated_data or _parse_response methods.
    3. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .

Args:
    model_series (str): Model series name identifier.
    embed_url (str): Embedding API URL address.
    api_key (str): API access key.
    embed_model_name (str): Embedding model name.
    return_trace (bool, optional): Whether to return trace information, defaults to False.


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
                 return_trace: bool = False,
                 batch_size: int = 10,
                 num_worker: int = 1,
                 timeout: int = 10):
        super().__init__(return_trace=return_trace)
        self._model_series = model_series
        self._embed_url = embed_url
        self._api_key = api_key
        self._embed_model_name = embed_model_name
        self._set_headers()
        self._batch_size = batch_size
        self._num_worker = num_worker
        self._timeout = timeout

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return 'EMBED'

    def _set_headers(self) -> Dict[str, str]:
        self._headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }

    def forward(self, input: Union[List, str], **kwargs) -> Union[List[float], List[List[float]]]:
        data = self._encapsulated_data(input, **kwargs)
        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        if isinstance(data, list):
            return self.run_embed_batch(input, data, proxies, **kwargs)
        else:
            with requests.post(self._embed_url, json=data, headers=self._headers, proxies=proxies,
                               timeout=self._timeout) as r:
                if r.status_code == 200:
                    return self._parse_response(r.json(), input)
                else:
                    raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))

    def _encapsulated_data(self, input: Union[List, str], **kwargs):
        if isinstance(input, str):
            json_data = {
                'input': [input],
                'model': self._embed_model_name
            }
            if len(kwargs) > 0:
                json_data.update(kwargs)
            return json_data
        else:
            text_batch = [input[i: i + self._batch_size] for i in range(0, len(input), self._batch_size)]
            json_data = [{'input': texts, 'model': self._embed_model_name} for texts in text_batch]
            if len(kwargs) > 0:
                for i in range(len(json_data)):
                    json_data[i].update(kwargs)
            return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> Union[List[List[float]], List[float]]:
        data = response.get('data', [])
        if not data:
            raise Exception('no data received')
        if isinstance(input, str):
            return data[0].get('embedding', [])
        else:
            return [res.get('embedding', []) for res in data]

    def run_embed_batch(self, input: List, data: List, proxies, **kwargs):
        """Internal method for executing batch embedding processing.

This method handles batch text embedding requests, supporting both single-threaded 
and multi-threaded processing modes. It automatically adjusts batch size and retries 
on request failures, providing robust error handling mechanisms.

Args:
    input (List): Original input text list
    data (List): Encapsulated batch request data list
    proxies: Proxy settings, set to None if NO_PROXY is True
    **kwargs: Additional keyword arguments


**Returns:**

- A list of embedding vector lists, each sublist corresponds to an input text's embedding vector

"""
        ret = [[] for _ in range(len(input))]
        flag = False
        if self._num_worker == 1:
            with requests.Session() as session:
                while not flag:
                    for i in range(len(data)):
                        r = session.post(self._embed_url, json=data[i], headers=self._headers,
                                         proxies=proxies, timeout=self._timeout)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input)
                            start = i * self._batch_size
                            ret[start: start + len(vec)] = vec
                            if i == len(data) - 1:
                                flag = True
                        else:
                            error_msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                            if self._batch_size == 1 or r.status_code in [401, 429]:
                                raise requests.RequestException(error_msg)
                            else:
                                msg = f'Online embedding:{self._embed_model_name} post failed, adjust batch_size: '
                                msg = msg + f' from {self._batch_size} to {max(self._batch_size // 2, 1)}'
                                LOG.warning(msg)
                                self._batch_size = max(self._batch_size // 2, 1)
                                data = self._encapsulated_data(input, **kwargs)
                                break
        else:
            with ThreadPoolExecutor(max_workers=self._num_worker) as executor:
                while not flag:
                    futures = [executor.submit(requests.post, self._embed_url, json=t, headers=self._headers,
                                               proxies=proxies, timeout=self._timeout) for t in data]
                    fut_to_index = {fut: idx for idx, fut in enumerate(futures)}
                    for fut in as_completed(futures):
                        r = fut.result()
                        i = fut_to_index.pop(fut)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input)
                            start = i * self._batch_size
                            ret[start: start + len(vec)] = vec
                            if len(fut_to_index) == 0:
                                flag = True
                        else:
                            wait(futures)
                            error_msg = '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                            if self._batch_size == 1 or r.status_code in [401, 429]:
                                raise requests.RequestException(error_msg)
                            else:
                                msg = f'Online embedding:{self._embed_model_name} post failed, adjust batch_size: '
                                msg = msg + f' from {self._batch_size} to {max(self._batch_size // 2, 1)}'
                                LOG.warning(msg)
                                self._batch_size = max(self._batch_size // 2, 1)
                                data = self._encapsulated_data(input, **kwargs)
                                break
        return ret
