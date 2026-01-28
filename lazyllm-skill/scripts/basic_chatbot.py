"""
基础对话机器人 - LazyLLM 基础示例

使用方法:
1. 设置环境变量: export LAZYLLM_QWEN_API_KEY=your_key
2. 运行: python basic_chatbot.py
"""

import lazyllm

# 创建对话模块
chat = lazyllm.OnlineModule()

# 设置提示词（可选）
prompt = '你是一个友好的助手，请用简洁明了的语言回答用户问题。'
chat.prompt(lazyllm.ChatPrompter(instruction=prompt))

# 对话循环
while True:
    query = input("query(enter 'quit' to exit): ")
    if query == "quit":
        break

    res = chat.forward(query)
    print(f"answer: {res}")
