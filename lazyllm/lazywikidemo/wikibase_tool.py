# wikibase_tools.py
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