"""
Web 界面集成示例

使用方法:
1. 设置环境变量: export LAZYLLM_QWEN_API_KEY=your_key
2. 运行: python web_interface.py
3. 访问浏览器: http://localhost:23333
"""

import lazyllm
from lazyllm.tools import FunctionCallAgent, fc_register

# 示例 1: 基础 Web 界面
print("示例 1: 基础 Web 界面")

chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat, port=23333, title='LazyLLM Chat').start().wait()

# 示例 2: 带提示词的 Web 界面
print("示例 2: 带提示词的 Web 界面")

chat = lazyllm.OnlineChatModule()
prompt = '你是一个专业的技术助手，请用准确、专业的语言回答技术问题。'
chat.prompt(lazyllm.ChatPrompter(instruction=prompt))
lazyllm.WebModule(chat, port=23334, title='Tech Assistant').start().wait()

# 示例 3: 带 Agent 的 Web 界面
print("示例 3: 带 Agent 的 Web 界面")

@fc_register('tool')
def my_tool(param: str) -> str:
    return f"工具处理结果: {param}"

llm = lazyllm.OnlineChatModule()
agent = FunctionCallAgent(llm, tools=['my_tool'])

lazyllm.WebModule(agent, port=23335, title='Agent Web').start().wait()

# 示例 4: 带静态资源的 Web 界面
print("示例 4: 带静态资源的 Web 界面")

chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(
    chat,
    port=23336,
    title='Chat with Images',
    static_paths='/path/to/images',
    enable_history=True
).start().wait()
