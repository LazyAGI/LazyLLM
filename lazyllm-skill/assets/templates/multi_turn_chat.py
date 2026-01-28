"""
多轮对话机器人 - 支持对话历史

使用方法:
1. 设置环境变量: export LAZYLLM_QWEN_API_KEY=your_key
2. 运行: python multi_turn_chat.py
"""

import lazyllm

# 创建对话模块
chat = lazyllm.OnlineChatModule()

# 对话历史: [[query1, answer1], [query2, answer2], ...]
history = []

print("多轮对话机器人（输入 'quit' 退出）")

while True:
    query = input("query: ")
    if query == "quit":
        break

    # 传递对话历史
    res = chat(query, llm_chat_history=history)
    print(f"answer: {res}\n")

    # 保存到历史
    history.append([query, res])
