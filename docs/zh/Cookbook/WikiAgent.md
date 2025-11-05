# 构建一个 Wikibase 工具 Agent

你是否希望你的 Agent 能够根据自然语言请求自动搜索 Wikidata 实体、属性，甚至运行 SPARQL 查询？在本节中，你将学习如何构建一个既懂中文又懂 Wikibase 的智能 Agent！

!!! abstract "在本节中，你将掌握以下 LazyLLM 核心知识"

- 如何封装并注册自定义工具以访问知识库和 SPARQL API；
- 如何让 Agent 根据名称解析 Q-ID 和 P-ID；
- 如何执行 SPARQL 查询并返回结果；
- 如何启动 ReactAgent 并提供网页服务。

---
## 设计思路
为了让我们的 AI 不仅能聊天，还能具备实时知识检索与事实查询能力，这里我们将引入 Wikidata 作为全球知识图谱数据库，让模型具备“查证事实、查询实体关系与属性”的能力。
我们将整合以下能力组件：
item_lookup：根据名称检索 Wikidata 实体并返回 Q-ID
property_lookup：根据属性名称检索 Wikidata 属性并返回 P-ID
sparql_query_runner：执行 SPARQL 查询以获取 Wikidata 中的结构化知识
OnlineChatModule：作为核心语言模型，理解问题并组织推理
ReactAgent：作为智能调度核心，让模型自动调用工具完成任务
我们注意到 Wikidata 查询分为实体识别 → 属性识别 → 查询执行三步，因此我们需要一个能够根据用户问题动态选择工具的智能体。另外，Wikidata 结构化查询返回 JSON 数据，需要模型解析与整合，因此我们让 LLM 根据需求主动发起多轮工具调用，然后汇总答案。
综合以上考虑，我们进行如下设计：
![Wikibase agent](../assets/wi.png)
## 三步构建 Wikibase Agent

问：如何让 LazyLLM 帮我处理实体/属性搜索和 SPARQL 查询？

答：只需三步！

1. 实现工具函数；
2. 使用 `@fc_register` 注册；
3. 启动 ReactAgent 和 Web 服务。

效果图示例：

![Wikibase Agent Demo](../assets/wikibase_agent_demo.png)

---

## 实现工具函数

以下是你构建 Wikibase 工具的典型代码结构。示例使用了 [Wikidata API](https://www.wikidata.org/w/api.php) 和 [SPARQL endpoint](https://query.wikidata.org/)。

```python
from thirdparty import httpx
from lazyllm import WebModule
from lazyllm.tools import fc_register
from lazyllm.module import OnlineChatModule
from lazyllm.tools.agent import ReactAgent

# 常量定义
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'lazyllm-agent/0.1 (zhangkejun@sensetime.com)'
}
```

---

### 工具函数：安全地提取嵌套 JSON

```python
def get_nested_value(o: dict, path: list) -> object:
    current = o
    for key in path:
        try:
            current = current[key]
        except (KeyError, TypeError):
            return None
    return current
```

---

### 工具 1：实体查找（Q-ID 查询）

```python
@fc_register('tool')
def item_lookup(search: str) -> str:
    '''
    Look up the Q-ID of a Wikidata item by its name.
    Args:
        search (str): The label or keyword of the entity to search in Wikidata.
    Returns:
        str: Q-ID of the entity (e.g., "Q1339") or error message.
    '''
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search,
        'srnamespace': 0,
        'srlimit': 1,
        'srqiprofile': 'classic_noboostlinks',
        'format': 'json'
    }
    response = httpx.get(WIKIDATA_API, params=params)
    title = get_nested_value(response.json(), ['query', 'search', 0, 'title'])
    return title.split(':')[-1] if title else f"I couldn't find any item for '{search}'"
```

---

### 工具 2：属性查找（P-ID 查询）

```python
@fc_register('tool')
def property_lookup(search: str) -> str:
    '''
    Look up the P-ID of a Wikidata property by its label.
    Args:
        search (str): The name of the property (e.g., "children", "instance of").
    Returns:
        str: P-ID of the property (e.g., "P40") or error message.
    '''
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search,
        'srnamespace': 120,
        'srlimit': 1,
        'srqiprofile': 'classic',
        'format': 'json'
    }
    response = httpx.get(WIKIDATA_API, params=params)
    title = get_nested_value(response.json(), ['query', 'search', 0, 'title'])
    return title.split(':')[-1] if title else f"I couldn't find any property for '{search}'"

```

---

### 工具 3：SPARQL 查询执行器

```python
@fc_register('tool')
def sparql_query_runner(query: str) -> str:
    '''
    Run a SPARQL query against Wikidata endpoint and return raw result.
    Args:
        query (str): SPARQL query string to execute.
    Returns:
        str: Raw JSON string of query result or error message.
    '''
    response = httpx.get(WIKIDATA_SPARQL, params={'query': query, 'format': 'json'})
    if response.status_code != 200:
        return 'That SPARQL query failed.'
    result = get_nested_value(response.json(), ['results', 'bindings'])
    return str(result)
```

---

## 启动 Agent 和 Web 服务

```python
if __name__ == '__main__':
    llm = OnlineChatModule()
    agent = ReactAgent(llm, tools=['item_lookup', 'property_lookup', 'sparql_query_runner'])
    WebModule(agent, port=range(23480, 23490)).start().wait()
```

---


```

---

## 查看完整代码
```python
from thirdparty import httpx
from lazyllm import WebModule
from lazyllm.tools import fc_register
from lazyllm.module import OnlineChatModule
from lazyllm.tools.agent import ReactAgent

# 常量定义
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'lazyllm-agent/0.1 (zhangkejun@sensetime.com)'
}


def get_nested_value(o: dict, path: list) -> object:
    current = o
    for key in path:
        try:
            current = current[key]
        except (KeyError, TypeError):
            return None
    return current


@fc_register('tool')
def item_lookup(search: str) -> str:
    '''
    Look up the Q-ID of a Wikidata item by its name.
    Args:
        search (str): The label or keyword of the entity to search in Wikidata.
    Returns:
        str: Q-ID of the entity (e.g., "Q1339") or error message.
    '''
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search,
        'srnamespace': 0,
        'srlimit': 1,
        'srqiprofile': 'classic_noboostlinks',
        'format': 'json'
    }
    response = httpx.get(WIKIDATA_API, params=params)
    title = get_nested_value(response.json(), ['query', 'search', 0, 'title'])
    return title.split(':')[-1] if title else f"I couldn't find any item for '{search}'"


@fc_register('tool')
def property_lookup(search: str) -> str:
    '''
    Look up the P-ID of a Wikidata property by its label.
    Args:
        search (str): The name of the property (e.g., "children", "instance of").
    Returns:
        str: P-ID of the property (e.g., "P40") or error message.
    '''
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': search,
        'srnamespace': 120,
        'srlimit': 1,
        'srqiprofile': 'classic',
        'format': 'json'
    }
    response = httpx.get(WIKIDATA_API, params=params)
    title = get_nested_value(response.json(), ['query', 'search', 0, 'title'])
    return title.split(':')[-1] if title else f"I couldn't find any property for '{search}'"


@fc_register('tool')
def sparql_query_runner(query: str) -> str:
    '''
    Run a SPARQL query against Wikidata endpoint and return raw result.
    Args:
        query (str): SPARQL query string to execute.
    Returns:
        str: Raw JSON string of query result or error message.
    '''
    response = httpx.get(WIKIDATA_SPARQL, params={'query': query, 'format': 'json'})
    if response.status_code != 200:
        return 'That SPARQL query failed.'
    result = get_nested_value(response.json(), ['results', 'bindings'])
    return str(result)


# --- 启动 Agent 和 Web 服务 ---
if __name__ == '__main__':
    llm = OnlineChatModule()
    agent = ReactAgent(llm, tools=['item_lookup', 'property_lookup', 'sparql_query_runner'])
    WebModule(agent, port=range(23480, 23490)).start().wait()

```
## 示例运行结果

示例输入：

```
Q: Q: What is the Q-ID for "Marie Curie"?
→ Returns Q7186
Q: What is the birth date of Albert Einstein?
→  Albert Einstein was born on March 14.1879.
---

这就完成啦 —— 现在你的 Agent 已经可以获取 Wikidata 实体、属性，并运行你提供的任何 SPARQL 查询了！

🎉 继续构建吧，让你的 AI 与这个世界的知识图谱紧密连接起来！