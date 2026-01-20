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

## 设计思路
首先考虑用UI-Web 接收来自用户的自然语言请求（例如“告诉我关于法国的信息”或“哪些国家使用美元？”），并将其发送给核心查询模块。该模块首先分析用户意图，识别出是查询“国家信息”还是“货币相关国家”，并自动提取关键实体（如国家名或货币代码）。

接着，系统根据识别出的意图和实体，动态构造对应的 REST API 请求 URL（例如 /v3.1/name/france 或 /v3.1/currency/USD），并调用外部 REST Countries API 获取结构化数据。

最后，系统将获取到的原始 JSON 数据返回给LLM Agent进行进一步处理与总结。
![alt text](../assets/api.png)
## 准备工作
获取API—KEY,具体过程详见：https://docs.lazyllm.ai/zh-cn/stable/Tutorial/2/#2-api-key
```python
pip install lazyllm
```
## 封装 API 工具类

我们先来看怎么构建一个可以自动识别问题并调用 API 的模块。以该API为例：https://restcountries.com/,REST Countries 是一个提供国家信息的开源API项目，用户可以通过RESTful接口获取关于国家的详细信息，如名称、首都、货币、语言等。

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
<details>
<summary>点击展开/折叠 Python代码</summary>

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
    """
    根据用户自然语言问题，自动识别意图、提取实体，并调用对应 API 端点。
    """
    def __init__(self, api_docs: str, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.base_url = self._extract_base_url(api_docs)
        self.endpoints = self._parse_endpoints(api_docs)

    def _extract_base_url(self, doc: str):
        """
        从 API 文档中提取 BASE URL。

        Args:
            doc (str): API 文档字符串。

        Returns:
            str: 提取出的基础 URL，末尾不带斜杠。
        """
        match = re.search(r'BASE URL:\s*(\S+)', doc)
        return match.group(1).rstrip("/") if match else ""

    def _parse_endpoints(self, doc: str):
        """
        从 API 文档中提取所有端点路径（带占位符的格式）。

        Args:
            doc (str): API 文档字符串。

        Returns:
            List[str]: 匹配到的端点路径列表，例如 ['/v3.1/currency/{currency}']。
        """
        pattern = r"The API endpoint\s+(/v[\d.]+/[^\s]+/\{[^}]+\})"
        return re.findall(pattern, doc)

    def _find_endpoint_for_question(self, question: str):
        """
        根据用户问题判断应调用哪个 API 端点，并提取所需参数。

        Args:
            question (str): 用户的自然语言问题。

        Returns:
            tuple: (endpoint_template, variables_dict)
                - endpoint_template: 带占位符的端点路径，如 "/v3.1/currency/{{currency}}"
                - variables_dict: 参数字典，如 {"currency": "USD"}

        Raises:
            ValueError: 无法识别问题意图时抛出。
        """
        q_lower = question.lower()
        # 如果问题涉及货币（包含 "currency" 或三个大写字母的货币代码）
        if "currency" in q_lower or re.search(r'\b[A-Z]{3}\b', question):
            code = self._extract_entity(question)
            return "/v3.1/currency/{{currency}}", {"currency": code}
        # 如果问题涉及国家信息（如国家名、首都、人口等）
        elif any(k in q_lower for k in ["country", "about", "information", "capital", "population"]):
            name = self._extract_entity(question)
            return "/v3.1/name/{{name}}", {"name": name}
        else:
            raise ValueError("无法识别问题所对应的 API endpoint")

    def _extract_entity(self, question: str):
        """
        从问题中提取关键实体（国家名或货币代码）。

        优先提取首字母大写的单词（如国家名），否则取最后一个单词。

        Args:
            question (str): 用户问题。

        Returns:
            str: 提取出的实体，默认为 "france"。
        """
        # 尝试匹配首字母大写的单词（如国家名）
        tokens = re.findall(r'\b[A-Z][a-z]+\b', question)
        if tokens:
            return tokens[-1]  # 取最后一个匹配项（避免误匹配如 "What"）
        # 否则从全小写单词中取最后一个（作为兜底）
        tokens = re.findall(r'\b[a-z]+\b', question.lower())
        return tokens[-1] if tokens else "france"

    def query(self, question: str):
        """
        主查询接口：接收自然语言问题，调用对应 API 并返回结果。

        Args:
            question (str): 用户的问题。

        Returns:
            str: API 返回的响应内容（JSON 字符串）。
        """
        # 根据问题确定端点和参数
        endpoint, variables = self._find_endpoint_for_question(question)
        # 拼接完整 URL（注意：此时 endpoint 中仍含 {{}} 占位符）
        url = self.base_url + endpoint

        # 创建 HTTP 请求对象（初始状态，占位符未替换）
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

        # 定义一个安全的 forward 方法，用于动态替换 URL/参数中的占位符
        def safe_forward(self, *args, **kwargs):
            """
            动态替换请求中的占位符（如 {{currency}}）为实际值。
            支持 URL、headers、params、body 中的变量替换。
            """
            def _map_input(target_str):
                """
                替换字符串中的 {{key}} 为 kwargs 或 args[0] 中对应的值。
                """
                if not isinstance(target_str, str):
                    return target_str
                # 合并 args 和 kwargs 中的变量
                replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
                if not replacements:
                    return target_str
                # 查找所有 {{xxx}} 形式的占位符
                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, target_str)
                for match in matches:
                    replacement = replacements.get(match)
                    if replacement is not None:
                        # 如果整个字符串就是占位符，直接返回值
                        if "{{" + match + "}}" == target_str:
                            return replacement
                        # 否则进行字符串替换（注意转义特殊字符）
                        target_str = re.sub(r"\{\{" + re.escape(match) + r"\}\}", replacement, target_str)
                return target_str

            # 替换 URL 中的占位符
            url = _map_input(self._url)
            # 替换 params 和 headers 中的占位符（如果存在）
            params = {key: _map_input(value) for key, value in self._params.items()} if self._params else None
            headers = {key: _map_input(value) for key, value in self._headers.items()} if self._headers else None
            # 处理 API Key（此处为 None，可能用于其他 API）
            headers, params = self._process_api_key(headers, params)

            # 判断是否为 JSON 请求
            if isinstance(headers, dict) and headers.get("Content-Type") == "application/json":
                try:
                    # 解析并替换 body 中的占位符
                    body = json.loads(self._body) if isinstance(self._body, str) else self._body
                    body = {k: _map_input(v) for k, v in body.items()}
                    http_response = httpx.request(
                        method=self._method, url=url, headers=headers,
                        params=params, json=body, timeout=self._timeout
                    )
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format: {self._body}")
            else:
                # 非 JSON 请求：处理 body 为字符串或字典
                if isinstance(self._body, dict):
                    body = json.dumps({k: _map_input(v) for k, v in self._body.items()})
                else:
                    body = _map_input(self._body)
                http_response = httpx.request(
                    method=self._method, url=url, headers=headers,
                    params=params, data=body, timeout=self._timeout
                )

            # 封装响应
            response = HttpExecutorResponse(http_response)
            _, file_binary = response.extract_file()
            # 如果响应包含文件二进制数据则返回 None，否则返回文本内容
            return response.content if len(file_binary) == 0 else None

        # 将 safe_forward 绑定到 request 实例上，作为其 forward 方法
        request.forward = safe_forward.__get__(request)
        # 调用 forward 并传入变量字典（如 {"currency": "USD"}）
        return request.forward(variables)


# 注册为函数调用工具，供 LLM Agent 使用
@fc_register
def query_restcountry(question: str) -> str:
    """
    对外暴露的工具函数：供 LLM 调用，查询 REST Countries API。

    Args:
        question (str): 用户关于国家或货币的自然语言问题。

    Returns:
        str: API 返回的 JSON 字符串结果。
    """
    return LazyAPIChain(api_docs=api_docs).query(question)


if __name__ == "__main__":
    # 初始化大语言模型（使用 Qwen 在线模型）
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])
    # 启动 Web 服务，监听 23480-23489 端口中的一个可用端口
    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()

```
</details>

---

至此，我们的第一个 API 交互 Agent 就大功告成啦！

🎉 让我们继续探索更多酷炫的 Agent 能力吧！
