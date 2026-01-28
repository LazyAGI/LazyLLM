"""
基础智能体 - 工具调用示例

使用方法:
1. 设置环境变量: export LAZYLLM_QWEN_API_KEY=your_key
2. 运行: python basic_agent.py
"""

from lazyllm import OnlineChatModule, WebModule
from lazyllm.tools import ReactAgent, fc_register

# 定义工具
@fc_register('tool')
def search_tool(query: str) -> str:
    """
    搜索工具，用于查找信息

    Args:
        query (str): 搜索查询

    Returns:
        str: 搜索结果
    """
    # 这里应该接入实际的搜索 API
    return f"搜索结果: {query}"

@fc_register('tool')
def calculate_tool(expression: str) -> str:
    """
    计算工具，用于执行数学运算

    Args:
        expression (str): 数学表达式

    Returns:
        str: 计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"

# 创建大模型
llm = OnlineChatModule()

# 创建智能体
agent = ReactAgent(llm, tools=['search_tool', 'calculate_tool'])

# 启动 Web 界面
print("启动智能体 Web 界面...")
WebModule(agent, port=12347, title='Basic Agent').start().wait()
