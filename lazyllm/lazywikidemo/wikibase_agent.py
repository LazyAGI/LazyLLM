# wikibase_server.py

from lazyllm.module import OnlineChatModule
from lazyllm.tools.agent import ReactAgent
from lazyllm import WebModule
from lazywikidemo.wikibase_tool import item_lookup, property_lookup, sparql_query_runner
from lazyllm.tools import fc_register  # 注册装饰器必须被显式导入一次才能生效

# --- 启动 Agent 和 Web 服务 ---
if __name__ == "__main__":
    # 初始化大模型模块（可改为 "gpt-4", "sensenova", "glm", "qwen", etc.）
    llm = OnlineChatModule(source="qwen", stream=False)

    # 构建 React Agent，并注册三种工具
    agent = ReactAgent(llm, tools=["ItemLookup", "PropertyLookup", "SparqlQueryRunner"])

    # 启动 Web API 接口服务（可被 POST /chat 调用）
    WebModule(agent, port=range(23480, 23490)).start().wait()
