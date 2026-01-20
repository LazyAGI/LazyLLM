# Build Your First API Interaction Agent

Want a large language model to automatically call real-world APIs based on your natural language questions? In this section, we’ll show you how to build a smart and action-capable Agent using LazyLLM!

!!! abstract "In this section, you'll learn the following LazyLLM essentials"

- How to inherit from [ModuleBase][lazyllm.module.module.ModuleBase] to implement an API tool module;
- How to wrap and register a function as a tool;
- How to use [ReactAgent][lazyllm.tools.agent.ReactAgent] with [WebModule][lazyllm.tools.WebModule] to enable API calling via Q&A.

---
## Design Approach
First, a UI-Web interface receives natural language requests from users (e.g., "Tell me about France" or "Which countries use the US dollar?") and forwards them to the core query module. This module analyzes the user's intent to determine whether the request pertains to "country information" or "countries using a specific currency," and automatically extracts key entities (such as country names or currency codes).

Next, based on the identified intent and extracted entities, the system dynamically constructs the appropriate REST API request URL (e.g., /v3.1/name/france or /v3.1/currency/USD) and calls the external REST Countries API to retrieve structured data.

Finally, the system returns the raw JSON response to an LLM Agent for further processing and summarization.
![alt text](../assets/api.png)
## Preparation
Obtain an API-KEY. For the detailed process, please refer to: https://docs.lazyllm.ai/zh-cn/stable/Tutorial/2/#2-api-key， REST Countries is an open-source API project that provides country information, allowing users to retrieve detailed data about countries—such as names, capitals, currencies, and languages—through RESTful interfaces.

```python
pip install lazyllm
```
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
<details> 
<summary>Click to expand full code</summary>
    
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
    It interprets natural language questions, identifies intent,
    extracts entities, and invokes the appropriate API endpoint.
    """

    def __init__(self, api_docs: str, verbose=False):
        """
        Initialize the LazyAPIChain.

        Args:
            api_docs (str): API documentation string containing BASE URL and endpoints.
            verbose (bool): Whether to enable verbose logging (currently unused, extensible).
        """
        super().__init__()
        self.verbose = verbose
        # Extract the base URL from the documentation
        self.base_url = self._extract_base_url(api_docs)
        # Parse all supported endpoint paths from the documentation
        self.endpoints = self._parse_endpoints(api_docs)

    def _extract_base_url(self, doc: str):
        """
        Extract the BASE URL from the API documentation.

        Args:
            doc (str): The API documentation string.

        Returns:
            str: The extracted base URL, with trailing slash removed.
        """
        match = re.search(r'BASE URL:\s*(\S+)', doc)
        return match.group(1).rstrip("/") if match else ""

    def _parse_endpoints(self, doc: str):
        """
        Extract all endpoint paths (with placeholders) from the documentation.

        Args:
            doc (str): The API documentation string.

        Returns:
            List[str]: A list of matched endpoint paths, e.g., ['/v3.1/currency/{currency}'].
        """
        # Match paths like /v3.1/xxx/{xxx}
        pattern = r"The API endpoint\s+(/v[\d.]+/[^\s]+/\{[^}]+\})"
        return re.findall(pattern, doc)

    def _find_endpoint_for_question(self, question: str):
        """
        Determine the appropriate API endpoint and parameters based on the user's question.

        Args:
            question (str): The user's natural language question.

        Returns:
            tuple: (endpoint_template, variables_dict)
                - endpoint_template: Endpoint path with double-brace placeholders, e.g., "/v3.1/currency/{{currency}}"
                - variables_dict: Parameter dictionary, e.g., {"currency": "USD"}

        Raises:
            ValueError: If the question intent cannot be recognized.
        """
        q_lower = question.lower()
        # Handle currency-related queries (contain "currency" or 3-letter uppercase codes like USD)
        if "currency" in q_lower or re.search(r'\b[A-Z]{3}\b', question):
            code = self._extract_entity(question)
            return "/v3.1/currency/{{currency}}", {"currency": code}
        # Handle country-related queries (e.g., about name, capital, population)
        elif any(k in q_lower for k in ["country", "about", "information", "capital", "population"]):
            name = self._extract_entity(question)
            return "/v3.1/name/{{name}}", {"name": name}
        else:
            raise ValueError("Unable to identify the corresponding API endpoint for the question.")

    def _extract_entity(self, question: str):
        """
        Extract the key entity (country name or currency code) from the question.

        Priority:
          1. Words starting with a capital letter (e.g., "France")
          2. Last lowercase word as fallback

        Args:
            question (str): The user's question.

        Returns:
            str: Extracted entity; defaults to "france" if none found.
        """
        # Try to match capitalized words (likely proper nouns like country names)
        tokens = re.findall(r'\b[A-Z][a-z]+\b', question)
        if tokens:
            return tokens[-1]  # Use the last match to avoid words like "What"
        # Fallback: use the last lowercase word
        tokens = re.findall(r'\b[a-z]+\b', question.lower())
        return tokens[-1] if tokens else "france"

    def query(self, question: str):
        """
        Main query interface: accepts a natural language question,
        calls the appropriate API, and returns the response.

        Args:
            question (str): User's natural language question.

        Returns:
            str: API response content (typically a JSON string).
        """
        # Determine endpoint template and parameters based on the question
        endpoint, variables = self._find_endpoint_for_question(question)
        # Construct full URL (still contains {{}} placeholders at this stage)
        url = self.base_url + endpoint

        # Create an HTTP request object (placeholders not yet substituted)
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

        # Define a safe `forward` method to dynamically substitute placeholders
        def safe_forward(self, *args, **kwargs):
            """
            Dynamically replace placeholders (e.g., {{currency}}) in URL, headers,
            params, and body with actual values from `kwargs` or `args[0]`.
            """
            def _map_input(target_str):
                """
                Replace {{key}} in a string with values from the provided variables.
                """
                if not isinstance(target_str, str):
                    return target_str
                # Merge variables from args (if dict) and kwargs
                replacements = {**kwargs, **(args[0] if args and isinstance(args[0], dict) else {})}
                if not replacements:
                    return target_str
                # Find all {{xxx}} placeholders
                pattern = r"\{\{([^}]+)\}\}"
                matches = re.findall(pattern, target_str)
                for match in matches:
                    replacement = replacements.get(match)
                    if replacement is not None:
                        # If the entire string is a placeholder, return the value directly
                        if "{{" + match + "}}" == target_str:
                            return replacement
                        # Otherwise, perform safe string substitution (escape regex special chars)
                        target_str = re.sub(r"\{\{" + re.escape(match) + r"\}\}", replacement, target_str)
                return target_str

            # Substitute placeholders in the URL
            url = _map_input(self._url)
            # Substitute placeholders in params and headers (if they exist)
            params = {key: _map_input(value) for key, value in self._params.items()} if self._params else None
            headers = {key: _map_input(value) for key, value in self._headers.items()} if self._headers else None
            # Process API key (currently None; placeholder for future extensibility)
            headers, params = self._process_api_key(headers, params)

            # Handle JSON requests
            if isinstance(headers, dict) and headers.get("Content-Type") == "application/json":
                try:
                    # Parse and substitute placeholders in JSON body
                    body = json.loads(self._body) if isinstance(self._body, str) else self._body
                    body = {k: _map_input(v) for k, v in body.items()}
                    http_response = httpx.request(
                        method=self._method, url=url, headers=headers,
                        params=params, json=body, timeout=self._timeout
                    )
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format: {self._body}")
            else:
                # Handle non-JSON requests (form data or raw body)
                if isinstance(self._body, dict):
                    body = json.dumps({k: _map_input(v) for k, v in self._body.items()})
                else:
                    body = _map_input(self._body)
                http_response = httpx.request(
                    method=self._method, url=url, headers=headers,
                    params=params, data=body, timeout=self._timeout
                )

            # Wrap the HTTP response
            response = HttpExecutorResponse(http_response)
            _, file_binary = response.extract_file()
            # Return content only if no binary file is detected
            return response.content if len(file_binary) == 0 else None

        # Bind the safe_forward method to the request instance
        request.forward = safe_forward.__get__(request)
        # Invoke the request with the extracted variables (e.g., {"currency": "USD"})
        return request.forward(variables)


# Register as a function-call tool for LLM agents
@fc_register
def query_restcountry(question: str) -> str:
    """
    Public tool function for LLMs to query country or currency information
    via the REST Countries API.

    Args:
        question (str): Natural language question about a country or currency.

    Returns:
        str: JSON-formatted API response string.
    """
    return LazyAPIChain(api_docs=api_docs).query(question)


# Main entry point: launch a web service for conversational country info queries
if __name__ == "__main__":
    # Initialize the LLM (using Qwen online model)
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["query_restcountry"])
    # Start a web server on an available port in the range 23480–23489
    lazyllm.WebModule(agent, port=range(23480, 23490)).start().wait()
```
</details>

---

That’s it — your first API interaction Agent is complete!

🎉 Let’s continue exploring more powerful Agent capabilities!
