from lazyllm.common import compile_func
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict, Any

class HttpTool(HttpRequest):
    def __init__(self,
                 method: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict[str, str]] = None,
                 code_str: Optional[str] = None,
                 vars_for_code: Optional[Dict[str, Any]] = None):
        super().__init__(method, url, '', headers, params, body)
        self._has_http = True if url else False
        self._compiled_func = compile_func(code_str, vars_for_code) if code_str else None

    def forward(self, *args, **kwargs):
        if self._has_http:
            res = super().forward(*args, **kwargs)
            return self._compiled_func(res) if self._compiled_func else res
        elif self._compiled_func:
            return self._compiled_func(*args, **kwargs)
        return None
