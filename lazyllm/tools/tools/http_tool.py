from lazyllm.common import compile_func, package
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict, Any, List
import json

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
                 vars_for_code: Optional[Dict[str, Any]] = None,
                 outputs: Optional[List[str]] = None,
                 extract_from_output: Optional[bool] = None):
        super().__init__(method, url, '', headers, params, body, timeout, proxies)
        self._has_http = True if url else False
        self._compiled_func = compile_func(code_str, vars_for_code) if code_str else (lambda x: json.loads(x['content']))
        self._outputs, self._extract_from_output = outputs, extract_from_output
        if extract_from_output:
            assert outputs, 'Output information is necessary to extract output parameters'

    def _get_result(self, res):
        if self._extract_from_output or (isinstance(res, dict) and len(self._outputs) > 1):
            assert isinstance(res, dict), 'The result of the tool should be a dict type'
            return package(res.get(key) for key in self._outputs)
        if len(self._outputs) > 1:
            assert isinstance(res, (tuple, list)), 'The result of the tool should be tuple or list'
            assert len(res) == len(self._outputs), 'The number of outputs is inconsistent with expectations'
            return package(res)
        return res

    def forward(self, *args, **kwargs):
        if self._has_http:
            res = super().forward(*args, **kwargs)
            if res['status_code'] != 200:
                raise RuntimeError('HttpRequest error')
            args, kwargs = (res,), {}
        res = self._compiled_func(*args, **kwargs)
        return self._get_result(res) if self._outputs else res
