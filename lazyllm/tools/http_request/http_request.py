import re

import httpx
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse

class HttpRequest(ModuleBase):
    def __init__(self, method, url, api_key, headers, params, body):
        super().__init__()
        self.method = method
        self.url = url
        self.api_key = api_key
        self.headers = headers
        self.params = params
        self.body = body
        self._process_api_key()

    def _process_api_key(self):
        if self.api_key != '':
            self.params['api_key'] = self.api_key

    def forward(self, *args, **kwargs):
        def _map_input(target_str):
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

        self.url = _map_input(self.url)
        self.body = _map_input(self.body)
        self.params = {key: _map_input(value) for key, value in self.params.items()}
        self.headers = {key: _map_input(value) for key, value in self.headers.items()}

        http_response = httpx.request(method=self.method, url=self.url, headers=self.headers,
                                      params=self.params, data=self.body)
        response = HttpExecutorResponse(http_response)

        _, file_binary = response.extract_file()

        outputs = {
            'status_code': response.status_code,
            'content': response.content if len(file_binary) == 0 else None,
            'headers': response.headers,
            'file': file_binary
        }
        return outputs
