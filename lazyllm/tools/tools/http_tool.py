from lazyllm.common import compile_func, package
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict, Any, List
import json

class HttpTool(HttpRequest):
    """
用于访问第三方服务和执行自定义代码的模块。参数中的 `params` 和 `headers` 的 value，以及 `body` 中可以包含形如 `{{variable}}` 这样用两个花括号标记的模板变量，然后在调用的时候通过参数来替换模板中的值。参考 [[lazyllm.tools.HttpTool.forward]] 中的使用说明。

Args:
    method (str, optional): 指定 http 请求方法，参考 `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`。
    url (str, optional): 要访问的 url。如果该字段为空，则表示该模块不需要访问第三方服务。
    params (Dict[str, str], optional): 请求 url 需要填充的 params 字段。如果 url 为空，该字段会被忽略。
    headers (Dict[str, str], optional): 访问 url 需要填充的 header 字段。如果 url 为空，该字段会被忽略。
    body (Dict[str, str], optional): 请求 url 需要填充的 body 字段。如果 url 为空，该字段会被忽略。
    timeout (int): 请求超时时间，单位是秒，默认值是 10。
    proxies (Dict[str, str], optional): 指定请求 url 时所使用的代理。代理格式参考 `https://www.python-httpx.org/advanced/proxies`。
    code_str (str, optional): 一个字符串，包含用户定义的函数。如果参数 `url` 为空，则直接执行该函数，执行时所有的参数都会转发给该函数；如果 `url` 不为空，该函数的参数为请求 url 返回的结果，此时该函数作为 url 返回后的后处理函数。
    vars_for_code (Dict[str, Any]): 一个字典，传入运行 code 所需的依赖及变量。



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
用于执行初始化时指定的操作：请求指定的 url 或者执行传入的函数。一般不直接调用，而是通过基类的 `__call__` 来调用。如果构造函数的 `url` 参数不为空，则传入的所有参数都会作为变量，用于替换在构造函数中使用 `{{}}` 标记的模板参数；如果构造函数的参数 `url` 为空，并且 `code_str` 不为空，则传入的所有参数都会作为 `code_str` 中所定义函数的参数。


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
