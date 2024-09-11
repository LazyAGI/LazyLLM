from lazyllm.common import compile_code
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict

class HttpTool(HttpRequest):
    def __init__(self,
                 method: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict[str, str]] = None,
                 py_code: Optional[str] = None):
        super().__init__(method, url, '', headers, params, body)
        self._has_http = True if url else False
        self._py_code = compile_code(py_code) if py_code else None

    def forward(self, *args, **kwargs):
        if self._has_http:
            res = super().forward(*args, **kwargs)
            return self._py_code(res) if self._py_code else res
        elif self._py_code:
            return self._py_code(*args, **kwargs)
        return None
