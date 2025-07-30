import re
import json
from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request.http_request import HttpRequest
from lazyllm import OnlineChatModule


class LazyAPIChain(ModuleBase):
    def __init__(self, llm, api_docs: str, verbose=False):
        super().__init__()
        self.llm = llm
        self.api_docs = api_docs
        self.verbose = verbose
        self.base_url = self._extract_base_url(api_docs)
        self.endpoints = self._parse_endpoints(api_docs)

    def _extract_base_url(self, doc: str):
        match = re.search(r'BASE URL:\s*(\S+)', doc)
        return match.group(1).rstrip("/") if match else ""

    def _parse_endpoints(self, doc: str):
        pattern = r"The API endpoint\s+(/v[\d.]+/[^{\s]+/\{[^}]+\})"
        return re.findall(pattern, doc)

    def _find_endpoint_for_question(self, question: str):
        q_lower = question.lower()
        if "currency" in q_lower or re.search(r"\b[A-Z]{3}\b", question):
            code = self._extract_entity(question)
            return "/v3.1/currency/{{currency}}", {"currency": code}
        elif "country" in q_lower or any(c in q_lower for c in ["france", "italy", "germany"]):
            name = self._extract_entity(question)
            return "/v3.1/name/{{name}}", {"name": name}
        else:
            raise ValueError("无法识别问题所对应的 API endpoint")

    def _extract_entity(self, question: str):
        tokens = re.findall(r'\b[A-Za-z]{2,}\b', question)
        return tokens[-1] if tokens else ""

    def run(self, question: str):
        if self.verbose:
            print("> Entering LazyAPIChain...")

        endpoint, variables = self._find_endpoint_for_question(question)
        url = self.base_url + endpoint

        if self.verbose:
            print(f"Request URL: {url}")
            print(f"With variables: {variables}")

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
            import httpx
            from lazyllm.tools.http_request.http_executor_response import HttpExecutorResponse

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
            return {
                'status_code': response.status_code,
                'content': response.content if len(file_binary) == 0 else None,
                'headers': response.headers,
                'file': file_binary
            }

        request.forward = safe_forward.__get__(request)

        result = request.forward(variables)

        raw_json = result.get("content", "")
        if isinstance(raw_json, bytes):
            raw_json = raw_json.decode("utf-8")

        if self.verbose:
            print("Raw JSON Response:", raw_json[:300], "...")

        prompt = f"Summarize the following country or currency data in plain English:\n{raw_json}\n"
        summary = self.llm(prompt)
        return summary


if __name__ == "__main__":
    llm_model = OnlineChatModule(source="qwen", stream=False)

    def llm(prompt):
        return llm_model(prompt)

    api_docs = """
    BASE URL: https://restcountries.com/

    The API endpoint /v3.1/name/{name} Used to find information about a country.
    The API endpoint /v3.1/currency/{currency} Used to find information about a region.
    """

    chain = LazyAPIChain(llm=llm, api_docs=api_docs, verbose=True)

    print("\n--- Example 1 ---")
    print(chain.run("Can you tell me information about France?"))

    print("\n--- Example 2 ---")
    print(chain.run("Tell me about the currency COP."))
