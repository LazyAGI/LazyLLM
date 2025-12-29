import copy
from contextlib import contextmanager
from typing import Dict, List, Union, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from lazyllm import LOG
from .utils import OnlineModuleBase


class OnlineEmbeddingModuleBase(OnlineModuleBase):
    NO_PROXY = True

    def __init__(self, model_series: str, embed_url: str, api_key: str, embed_model_name: str,
                 return_trace: bool = False, batch_size: int = 1, num_worker: int = 1, timeout: int = 10):
        super().__init__(api_key=api_key, skip_auth=False, return_trace=return_trace)
        self._model_series = model_series
        self._embed_url = embed_url
        self._embed_model_name = embed_model_name
        self._batch_size = batch_size
        self._num_worker = num_worker
        self._timeout = timeout
        if hasattr(self, '_set_embed_url'): self._set_embed_url()

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return 'EMBED'

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    def share(self, *, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None):
        new = copy.copy(self)
        if embed_url is not None:
            new._embed_url = embed_url
        if embed_model_name is not None:
            new._embed_model_name = embed_model_name
        return new

    @contextmanager
    def _override_endpoint(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None):
        old_url = self._embed_url
        old_model = self._embed_model_name
        if embed_url is not None:
            self._embed_url = embed_url
        if embed_model_name is not None:
            self._embed_model_name = embed_model_name
        try:
            yield
        finally:
            self._embed_url = old_url
            self._embed_model_name = old_model

    def forward(self, input: Union[List, str], **kwargs) -> Union[List[float], List[List[float]]]:
        runtime_embed_url = kwargs.pop('embed_url', kwargs.pop('base_url', None))
        runtime_embed_model = kwargs.pop('embed_model_name', kwargs.pop('model_name', None))
        override_model = kwargs.get('model', runtime_embed_model)
        with self._override_endpoint(runtime_embed_url, override_model):
            data = self._encapsulated_data(input, **kwargs)
            proxies = {'http': None, 'https': None} if self.NO_PROXY else None
            if isinstance(data, list):
                return self.run_embed_batch(input, data, proxies, **kwargs)
            else:
                with requests.post(self._embed_url, json=data, headers=self._header, proxies=proxies,
                                   timeout=self._timeout) as r:
                    if r.status_code == 200:
                        return self._parse_response(r.json(), input=input)
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
        ret = [[] for _ in range(len(input))]
        flag = False
        if self._num_worker == 1:
            with requests.Session() as session:
                while not flag:
                    for i in range(len(data)):
                        r = session.post(self._embed_url, json=data[i], headers=self._header,
                                         proxies=proxies, timeout=self._timeout)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input=input)
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
                    futures = [executor.submit(requests.post, self._embed_url, json=t, headers=self._header,
                                               proxies=proxies, timeout=self._timeout) for t in data]
                    fut_to_index = {fut: idx for idx, fut in enumerate(futures)}
                    for fut in as_completed(futures):
                        r = fut.result()
                        i = fut_to_index.pop(fut)
                        if r.status_code == 200:
                            vec = self._parse_response(r.json(), input=input)
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
