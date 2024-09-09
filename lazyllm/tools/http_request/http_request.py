import re

import httpx
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse

class HttpRequest(ModuleBase):
    def __init__(self, method, url, api_key, headers, params, body, timeout=10, proxies=None):
        super().__init__()
        if not url:
            return

        self._method = method
        self._url = url
        self._api_key = api_key
        self._headers = headers
        self._params = params
        self._body = body
        self._timeout = timeout
        self._proxies = proxies
        self._process_api_key()

    def _process_api_key(self):
        if self._api_key and self._api_key != '':
            self._params['api_key'] = self._api_key

    def forward(self, *args, **kwargs):
        def _map_input(target_str):
            if not isinstance(target_str, str):
                return target_str

            # TODO: replacements could be more complex to create.
            replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
            if not replacements:
                return target_str

            pattern = r"\{\{([^}]+)\}\}"
            matches = re.findall(pattern, target_str)
            for match in matches:
                replacement = replacements.get(match)
                if replacement is not None:
                    target_str = re.sub(r"\{\{" + re.escape(match) + r"\}\}", replacement, target_str)

            return target_str

        self._url = _map_input(self._url)

        if self._body:
            self._body = _map_input(self._body)
        if self._params:
            self._params = {key: _map_input(value) for key, value in self._params.items()}
        if self._headers:
            self._headers = {key: _map_input(value) for key, value in self._headers.items()}

        http_response = httpx.request(method=self._method, url=self._url, headers=self._headers,
                                      params=self._params, data=self._body, timeout=self._timeout,
                                      proxies=self._proxies)
        response = HttpExecutorResponse(http_response)

        _, file_binary = response.extract_file()

        outputs = {
            'status_code': response.status_code,
            'content': response.content if len(file_binary) == 0 else None,
            'headers': response.headers,
            'file': file_binary
        }
        return outputs
