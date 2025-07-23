from lazyllm.common import compile_func, package
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict, Any, List
import json

class HttpTool(HttpRequest):
    """
Module for accessing third-party services and executing custom code. The values in `params` and `headers`, as well as in body, can include template variables marked with double curly braces like `{{variable}}`, which are then replaced with actual values through parameters when called. Refer to the usage instructions in [[lazyllm.tools.HttpTool.forward]].

Args:
    method (str, optional): Specifies the HTTP request method, refer to `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`.
    url (str, optional): The URL to access. If this field is empty, it indicates that the module does not need to access third-party services.
    params (Dict[str, str], optional): Params fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    headers (Dict[str, str], optional): Header fields to be filled when accessing the URL. If the URL is empty, this field will be ignored.
    body (Dict[str, str], optional): Body fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    timeout (int): Request timeout in seconds, default value is 10.
    proxies (Dict[str, str], optional): Specifies the proxies to be used when requesting the URL. Proxy format refer to `https://www.python-httpx.org/advanced/proxies`.
    code_str (str, optional): A string containing a user-defined function. If the parameter url is empty, execute this function directly, forwarding all arguments to it; if url is not empty, the parameters of this function are the results returned from the URL request, and in this case, the function serves as a post-processing function for the URL response.
    vars_for_code (Dict[str, Any]): A dictionary that includes dependencies and variables required for running the code.



Examples:
    
    from lazyllm.tools import HttpTool
    
    code_str = "def identity(content): return content"
    tool = HttpTool(method='GET', url='http://www.sensetime.com/', code_str=code_str)
    ret = tool()
    """
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
                 extract_from_result: Optional[bool] = None):
        super().__init__(method, url, '', headers, params, body, timeout, proxies)
        self._has_http = True if url else False
        self._compiled_func = (compile_func(code_str, vars_for_code) if code_str else
                               (lambda x: json.loads(x['content'])) if self._has_http else None)
        self._outputs, self._extract_from_result = outputs, extract_from_result
        if extract_from_result:
            assert outputs, 'Output information is necessary to extract output parameters'
            assert len(outputs) == 1, 'When the number of outputs is greater than 1, no manual setting is required'

    def _get_result(self, res):
        if self._extract_from_result or (isinstance(res, dict) and len(self._outputs) > 1):
            assert isinstance(res, dict), 'The result of the tool should be a dict type'
            r = package(res.get(key) for key in self._outputs)
            return r[0] if len(r) == 1 else r
        if len(self._outputs) > 1:
            assert isinstance(res, (tuple, list)), 'The result of the tool should be tuple or list'
            assert len(res) == len(self._outputs), 'The number of outputs is inconsistent with expectations'
            return package(res)
        return res

    def forward(self, *args, **kwargs):
        """
Used to perform operations specified during initialization: request the specified URL or execute the passed function. Generally not called directly, but through the base class's `__call__`. If the `url` parameter in the constructor is not empty, all passed parameters will be used as variables to replace template parameters marked with `{{}}` in the constructor; if the `url` parameter in the constructor is empty and `code_str` is not empty, all passed parameters will be used as arguments for the function defined in `code_str`.


Examples:
    
    from lazyllm.tools import HttpTool
    
    code_str = "def exp(v, n): return v ** n"
    tool = HttpTool(code_str=code_str)
    assert tool(v=10, n=2) == 100
    """
        if not self._compiled_func: return None
        if self._has_http:
            res = super().forward(*args, **kwargs)
            if int(res['status_code']) >= 400:
                raise RuntimeError(f'HttpRequest error, status code is {res["status_code"]}.')
            args, kwargs = (res,), {}
        res = self._compiled_func(*args, **kwargs)
        return self._get_result(res) if self._outputs else res
