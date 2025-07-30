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

api_docs = """
BASE URL: https://restcountries.com/

The API endpoint /v3.1/name/{name} Used to find information about a country.
The API endpoint /v3.1/currency/{currency} Used to find information about a region.
"""

# --- Tool 封装类 ---
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

# --- Tool 注册 ---
@fc_register
def query_restcountry(question: str) -> str:
    '''
    Query country or currency information based on the user question.

    Args:
        question (str): User question to identify which endpoint to call.
    '''
    return LazyAPIChain(api_docs=api_docs).query(question)

# --- 启动 Agent 和 Web 服务 ---
if __name__ == "__main__":
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])

    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()