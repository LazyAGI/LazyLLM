import re
import copy
from lazyllm.components.http_request.http_executor import HttpExecutor, HttpExecutorResponse


class HttpRequestNode():
    def __init__(self, method, url, API_Key, headers, params, body):
        self.method = method
        self.url = url
        self.API_Key = API_Key
        self.headers = headers
        self.params = params
        self.body = body
        self._process_api_key()

    def _process_api_key(self):
        if self.API_Key != '':
            self.params['api_key'] = self.API_Key

    def __call__(self, *args, **kwargs):
        def _map_input(target_str):
            # TODO: replacements could be more complex to create.
            replacements = copy.deepcopy(kwargs)
            if len(args) > 0 and isinstance(args[0], dict):
                replacements.update(args[0])

            if len(replacements) == 0:
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
        for key, value in self.params.items():
            self.params[key] = _map_input(value)

        for key, value in self.headers.items():
            self.headers[key] = _map_input(value)

        try:
            http_executor = HttpExecutor(self.method, self.url, self.headers, self.params, self.body)
            response = http_executor.invoke()
        except Exception as e:
            raise e

        _, file_binary = self.extract_files(response)

        outputs = {'status_code': response.status_code,
                   'content': response.content if len(file_binary) == 0 else None,
                   'headers': response.headers,
                   'file': file_binary
                   }
        return outputs

    def extract_files(self, response: HttpExecutorResponse):
        return response.extract_file()
