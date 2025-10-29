# Build a Wikibase Tools Agent

Do you want your Agent to search for Wikidata entities, properties, and even run SPARQL queries automatically based on your natural language requests? In this section, you’ll learn how to build an Agent that speaks both English and Wikibase!

!!! abstract "In this section, you’ll master these LazyLLM essentials"

- How to wrap and register custom tools for knowledge base and SPARQL API access;
- How to let an Agent resolve Q-IDs and P-IDs by name;
- How to run SPARQL queries and return results to users;
- How to launch a ReactAgent and serve it on the web.

---

## Three Steps to Build a Wikibase Agent

Q: How do I make LazyLLM handle entity/property search and SPARQL queries for me?

A: Just three steps!

1. Implement the tool functions;
2. Register them with `@fc_register`;
3. Launch your ReactAgent and web service.

Result preview:

![Wikibase Agent Demo](../assets/wikibase_agent_demo.png)

---

## Implement the Tool Functions

Here is a typical code layout for your Wikibase tools. The example uses the [Wikidata API](https://www.wikidata.org/w/api.php) and [SPARQL endpoint](https://query.wikidata.org/).

```python
import httpx
from lazyllm.tools import fc_register
from lazyllm.module import OnlineChatModule
from lazyllm.tools.agent import ReactAgent
from lazyllm import WebModule
from typing import Optional, Dict, Any

# Constant definitions
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "lazyllm-agent/0.1 (contact@example.com)"  # Replace with your own info
}
```

---

### Helper: Safe Nested JSON Extraction

```python
def get_nested_value(o: Dict, path: list) -> Any:
    current = o
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current
```

---

### Tool 1: Entity Lookup (Q-ID Search)

```python
@fc_register("tool")
def item_lookup(search: str) -> str:
    '''
    Look up the Q-ID of a Wikidata item by its name.
    '''
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": 0,
        "srlimit": 1,
        "srqiprofile": "classic_noboostlinks",
        "format": "json"
    }
    response = httpx.get("https://www.wikidata.org/w/api.php", params=params)
    title = get_nested_value(response.json(), ["query", "search", 0, "title"])
    return title.split(":")[-1] if title else f"I couldn't find any item for '{search}'"
```

---

### Tool 2: Property Lookup (P-ID Search)

```python
@fc_register("tool")
def property_lookup(search: str) -> str:
    '''
    Look up the P-ID of a Wikidata property by its label.
    '''
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": 120,
        "srlimit": 1,
        "srqiprofile": "classic",
        "format": "json"
    }
    response = httpx.get("https://www.wikidata.org/w/api.php", params=params)
    title = get_nested_value(response.json(), ["query", "search", 0, "title"])
    return title.split(":")[-1] if title else f"I couldn't find any property for '{search}'"
```

---

### Tool 3: SPARQL Query Runner

```python
@fc_register("tool")
def sparql_query_runner(query: str) -> str:
    '''
    Run a SPARQL query against Wikidata endpoint and return raw result.
    '''
    response = httpx.get("https://query.wikidata.org/sparql", params={
        "query": query,
        "format": "json"
    })
    if response.status_code != 200:
        return "That SPARQL query failed."
    result = get_nested_value(response.json(), ["results", "bindings"])
    return str(result)
```

---

## Launch the Agent and Web Service

```python
if __name__ == "__main__":
    llm = OnlineChatModule(source="qwen", stream=False)
    agent = ReactAgent(llm, tools=["item_lookup", "property_lookup", "sparql_query_runner"])
    WebModule(agent, port=range(23480, 23490)).start().wait()
```

---

## Try It Out

Example inputs:

```
Q: What is the Q-ID for "Marie Curie"?
→ Returns Q7186
Q: What is the birth date of Albert Einstein?
→  Albert Einstein was born on March 14.1879.
```

---

## View Full Code

```python
import httpx
from lazyllm.tools import fc_register
from lazyllm.module import OnlineChatModule
from lazyllm.tools.agent import ReactAgent
from lazyllm import WebModule
from typing import Optional, Dict, Any

# 常量定义
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "lazyllm-agent/0.1 (contact@example.com)"  # 替换为你自己的邮箱/信息
}


def get_nested_value(o: Dict, path: list) -> Any:
    current = o
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current


@fc_register("tool")
def item_lookup(search: str) -> str:
    '''
    Look up the Q-ID of a Wikidata item by its name.
    Args:
        search (str): The label or keyword of the entity to search in Wikidata.
    Returns:
        str: Q-ID of the entity (e.g., "Q1339") or error message.
    '''
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": 0,
        "srlimit": 1,
        "srqiprofile": "classic_noboostlinks",
        "format": "json"
    }
    response = httpx.get("https://www.wikidata.org/w/api.php", params=params)
    title = get_nested_value(response.json(), ["query", "search", 0, "title"])
    return title.split(":")[-1] if title else f"I couldn't find any item for '{search}'"


@fc_register("tool")
def property_lookup(search: str) -> str:
    '''
    Look up the P-ID of a Wikidata property by its label.
    Args:
        search (str): The name of the property (e.g., "children", "instance of").
    Returns:
        str: P-ID of the property (e.g., "P40") or error message.
    '''
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": 120,
        "srlimit": 1,
        "srqiprofile": "classic",
        "format": "json"
    }
    response = httpx.get("https://www.wikidata.org/w/api.php", params=params)
    title = get_nested_value(response.json(), ["query", "search", 0, "title"])
    return title.split(":")[-1] if title else f"I couldn't find any property for '{search}'"


@fc_register("tool")
def sparql_query_runner(query: str) -> str:
    '''
    Run a SPARQL query against Wikidata endpoint and return raw result.
    Args:
        query (str): SPARQL query string to execute.
    Returns:
        str: Raw JSON string of query result or error message.
    '''
    response = httpx.get("https://query.wikidata.org/sparql", params={
        "query": query,
        "format": "json"
    })
    if response.status_code != 200:
        return "That SPARQL query failed."
    result = get_nested_value(response.json(), ["results", "bindings"])
    return str(result)

# --- 启动 Agent 和 Web 服务 ---
if __name__ == "__main__":
    # 初始化大模型模块（可改为 "gpt-4", "sensenova", "glm", "qwen", etc.）
    llm = OnlineChatModule(source="qwen", stream=False)

    # 构建 React Agent，并注册三种工具
    agent = ReactAgent(llm, tools=["item_lookup", "property_lookup", "sparql_query_runner"])

    # 启动 Web API 接口服务（可被 POST /chat 调用）
    WebModule(agent, port=range(23480, 23490)).start().wait()
```

---

That’s it — now your Agent can fetch Wikidata entities, properties, and run any SPARQL you throw at it!

🎉 Keep building and connecting your AI with the world’s knowledge graph!