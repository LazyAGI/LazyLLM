# 构建你的第一个 API 交互 Agent

想让大模型根据你的自然语言问题自动调用真实世界的 API 吗？这节我们就来教你，如何用 LazyLLM 打造这样一个“聪明又动手能力强”的 Agent！

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

- 如何继承 [ModuleBase][lazyllm.module.module.ModuleBase] 编写一个 API 工具模块；
- 如何封装并注册一个函数为工具；
- 如何使用 [ReactAgent][lazyllm.tools.agent.ReactAgent] 与 [WebModule][lazyllm.tools.WebModule] 联动，实现问答驱动的 API 调用。

---

## 三步构建 API Agent

问：用 LazyLLM 构建一个自动调用 REST API 的 Agent，总共分几步？

答：三步！

1. 封装 API 工具；
2. 注册为 Agent 工具；
3. 启动 React Agent 和客户端。

效果图如下：

![API Agent Demo](../assets/api_agent_demo.png)

---
## 准备工作
获取API—KEY,具体过程详见：https://docs.lazyllm.ai/zh-cn/stable/Tutorial/2/#2-api-key
```python
pip install lazyllm
```
## 封装 API 工具类

我们先来看怎么构建一个可以自动识别问题并调用 API 的模块。以该API为例：https://restcountries.com/

```python
class LazyAPIChain(ModuleBase):
    def __init__(self, api_docs: str, verbose=False):
        ...
```

- 我们继承了 `ModuleBase`，这是所有工具模块的基类；
- 传入的 `api_docs` 是一段 REST API 文档字符串，会自动解析出 Base URL 和 Endpoint 模板；
- 模块会根据用户的问题智能判断该调用哪个 endpoint，并自动完成参数抽取和请求发送。

### 自动匹配接口

```python
def _find_endpoint_for_question(self, question: str):
    ...
```

- 通过关键词判断问题是查“国家”还是查“货币”；
- 例如 “What is the population of Germany?” → 会选中 `/v3.1/name/{name}`；
- 再通过 `_extract_entity` 方法提取出参数（如 `Germany`）。

---

### 构造 HttpRequest

```python
request = HttpRequest(
    method="GET",
    url=url,
    headers={},
    ...
)
```

- 我们使用 LazyLLM 内置的 `HttpRequest` 类来构造请求；
- 它支持变量占位（如 `{{currency}}`），后续会通过 `safe_forward` 自动替换；
- 最终将调用真实 API 并返回结果。

---

## 注册为工具函数

问：Agent 怎么知道你写的这个工具函数能干嘛？

答：用 `@fc_register` 把它暴露出去！

```python
@fc_register
def query_restcountry(question: str) -> str:
    '''
    Query country or currency information based on the user question.
    '''
    return LazyAPIChain(api_docs=api_docs).query(question)
```

- 函数名 `query_restcountry` 就是工具的调用名；
- `question: str` 是它的输入说明；
- 内部调用我们刚刚封装好的工具类。

---

## 启动 Agent 和客户端

```python
if __name__ == "__main__":
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])
    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()
```

- 用 `OnlineChatModule` 启动一个大模型对话模块；
- 用 `ReactAgent` 构建一个具备推理和函数调用能力的 Agent；
- 用 `WebModule` 启动客户端服务。

成功后你会看到类似这样的终端输出：

```arduino
服务已启动：http://localhost:23480
```

你就可以打开网页开始提问了！

---

## 效果展示

让我们输入一些问题试试吧：

```
Q: What is the population of France?
→ 会自动调用 /v3.1/name/{name} 接口，返回人口信息。

Q: What countries use USD?
→ 会自动调用 /v3.1/currency/{currency} 接口，返回相关国家。
```

是不是很神奇？我们的 API Agent 不仅能看懂你的问题，还能“动手”去网上找答案！

---

## 查看完整代码

```python
import re
import json
import httpx
import lazyllm
from typing import List, Dict, Any
from lazyllm import OnlineChatModule
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_request import HttpRequest
from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse
from lazyllm.tools.agent import ReactAgent
from lazyllm.tools import fc_register

api_docs = '''
BASE URL: https://restcountries.com/

The API endpoint /v3.1/name/{name} Used to find information about a country.
The API endpoint /v3.1/currency/{currency} Used to find information about a region.
'''

class LazyAPIChain(ModuleBase):
    def __init__(self, api_docs: str, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.base_url = self._extract_base_url(api_docs)
        self.endpoints = self._parse_endpoints(api_docs)

    def _extract_base_url(self, doc: str):
        match = re.search(r'BASE URL:\s*(\S+)', doc)
        return match.group(1).rstrip("/") if match else ""

    def _parse_endpoints(self, doc: str):
        pattern = r"The API endpoint\s+(/v[\d.]+/[^\s]+/\{[^}]+\})"
        return re.findall(pattern, doc)

    def _find_endpoint_for_question(self, question: str):
        q_lower = question.lower()
        if "currency" in q_lower or re.search(r'\b[A-Z]{3}\b', question):
            code = self._extract_entity(question)
            return "/v3.1/currency/{{currency}}", {"currency": code}
        elif any(k in q_lower for k in ["country", "about", "information", "capital", "population"]):
            name = self._extract_entity(question)
            return "/v3.1/name/{{name}}", {"name": name}
        else:
            raise ValueError("无法识别问题所对应的 API endpoint")

    def _extract_entity(self, question: str):
        tokens = re.findall(r'\b[A-Z][a-z]+\b', question)
        if tokens:
            return tokens[-1]
        tokens = re.findall(r'\b[a-z]+\b', question.lower())
        return tokens[-1] if tokens else "france"

    def query(self, question: str):
        endpoint, variables = self._find_endpoint_for_question(question)
        url = self.base_url + endpoint

        request = HttpRequest(
            method="GET",
            url=url,
            api_key=None,
            headers={},
            params={},
            body=None,
            timeout=10,
            proxies=None
        )

        def safe_forward(self, *args, **kwargs):
            def _map_input(target_str):
                if not isinstance(target_str, str):
                    return target_str
                replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
                if not replacements:
                    return target_str
                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, target_str)
                for match in matches:
                    replacement = replacements.get(match)
                    if replacement is not None:
                        if "{{" + match + "}}" == target_str:
                            return replacement
                        target_str = re.sub(r"\{\{" + re.escape(match) + r"\}\}", replacement, target_str)
                return target_str

            url = _map_input(self._url)
            params = {key: _map_input(value) for key, value in self._params.items()} if self._params else None
            headers = {key: _map_input(value) for key, value in self._headers.items()} if self._headers else None
            headers, params = self._process_api_key(headers, params)

            if isinstance(headers, dict) and headers.get("Content-Type") == "application/json":
                try:
                    body = json.loads(self._body) if isinstance(self._body, str) else self._body
                    body = {k: _map_input(v) for k, v in body.items()}
                    http_response = httpx.request(method=self._method, url=url, headers=headers,
                                                      params=params, json=body, timeout=self._timeout)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format: {self._body}")
            else:
                body = (json.dumps({k: _map_input(v) for k, v in self._body.items()})
                        if isinstance(self._body, dict) else _map_input(self._body))
                http_response = httpx.request(method=self._method, url=url, headers=headers,
                                                params=params, data=body, timeout=self._timeout)

            response = HttpExecutorResponse(http_response)
            _, file_binary = response.extract_file()
            return response.content if len(file_binary) == 0 else None

        request.forward = safe_forward.__get__(request)
        return request.forward(variables)

@fc_register
def query_restcountry(question: str) -> str:
    return LazyAPIChain(api_docs=api_docs).query(question)

if __name__ == "__main__":
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])
    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()
```

---

至此，我们的第一个 API 交互 Agent 就大功告成啦！

🎉 让我们继续探索更多酷炫的 Agent 能力吧！
