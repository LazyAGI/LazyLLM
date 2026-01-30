import lazyllm

# 基础对话chatbot
# 创建对话模块
chat = lazyllm.OnlineModule()

# 设置提示词
prompt = '你是一个友好的助手，请用简洁明了的语言回答用户问题。'
chat.prompt(lazyllm.ChatPrompter(instruction=prompt))

# 对话循环
while True:
    query = input("query(enter 'quit' to exit): ")
    if query == 'quit':
        break

    res = chat.forward(query)
    print(f'answer: {res}')

# 添加对话历史
history = []
print("多轮对话机器人（输入 'quit' 退出）")
while True:
    query = input('query: ')
    if query == 'quit':
        break

    # 传递对话历史
    res = chat(query, llm_chat_history=history)
    print(f'answer: {res}\n')
    # 保存到历史
    history.append([query, res])

# 优化历史对话添加
chat = lazyllm.TrainableModule('internlm2-chat-7b')
lazyllm.WebModule(chat, port=23466, history=[chat]).start().wait()
