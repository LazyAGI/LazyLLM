# Build Your First API Interaction Agent

Want a large language model to automatically call real-world APIs based on your natural language questions? In this section, we’ll show you how to build a smart and action-capable Agent using LazyLLM!

!!! abstract "In this section, you'll learn the following LazyLLM essentials"

- How to inherit from [ModuleBase][lazyllm.module.module.ModuleBase] to implement an API tool module;
- How to wrap and register a function as a tool;
- How to use [ReactAgent][lazyllm.tools.agent.ReactAgent] with [WebModule][lazyllm.tools.WebModule] to enable API calling via Q&A.

---

## Build an API Agent in Three Steps

Q: How many steps does it take to build an API-calling Agent using LazyLLM?

A: Three steps!

1. Wrap the API as a tool;
2. Register it as an Agent tool;
3. Launch the ReactAgent and web client.

Here’s the result:

![API Agent Demo](../assets/api_agent_demo.png)

---

## Wrap the API as a Tool Class

Let’s first look at how to build a module that can intelligently recognize questions and call APIs. We'll use the API at: https://restcountries.com/

```python
class LazyAPIChain(ModuleBase):
    def __init__(self, api_docs: str, verbose=False):
        ...
```

- We inherit from `ModuleBase`, the base class for all LazyLLM modules;
- The input `api_docs` is a REST API documentation string that’s parsed to extract the base URL and endpoint templates;
- The module determines which endpoint to call based on the user’s question, extracts parameters, and sends the request.

### Automatically Match Endpoints

```python
def _find_endpoint_for_question(self, question: str):
    ...
```

- Determine whether the question is about "country" or "currency" using keyword detection;
- For example: “What is the population of Germany?” → it will match `/v3.1/name/{name}`;
- `_extract_entity` is used to extract the parameter (e.g., `Germany`).

---

### Build the HttpRequest

```python
request = HttpRequest(
    method="GET",
    url=url,
    headers={},
    ...
)
```

- We use LazyLLM’s built-in `HttpRequest` class to construct the request;
- It supports variable placeholders like `{{currency}}`, which will be filled in via `safe_forward`;
- The request is sent to the real API and the result is returned.

---

## Register as a Tool Function

Q: How does the Agent know what your tool function does?

A: Use `@fc_register` to expose it!

```python
@fc_register
def query_restcountry(question: str) -> str:
    '''
    Query country or currency information based on the user question.
    '''
    return LazyAPIChain(api_docs=api_docs).query(question)
```

- The function name `query_restcountry` becomes the tool name;
- `question: str` defines the input interface;
- Inside, we call the API tool class we just created.

---

## Launch the Agent and Web Client

```python
if __name__ == "__main__":
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])
    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()
```

- Use `OnlineChatModule` to launch a model-based dialogue module;
- Use `ReactAgent` to construct an agent with reasoning and function-calling capabilities;
- Use `WebModule` to serve a web-based client.

Once successful, you’ll see output like:

```arduino
Service started: http://localhost:23480
```

You can now open the webpage and start asking questions!

---

## Demo Showcase

Let’s try a few questions:

```
Q: What is the population of France?
→ Will automatically call /v3.1/name/{name} and return population data.

Q: What countries use USD?
→ Will automatically call /v3.1/currency/{currency} and return matching countries.
```

Isn’t it amazing? Our API Agent not only understands your question but also “gets the job done” by calling external services!

---

## View Full Code


    
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

That’s it — your first API interaction Agent is complete!

🎉 Let’s continue exploring more powerful Agent capabilities!
