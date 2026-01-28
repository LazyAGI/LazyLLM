"""
基础 RAG 知识库问答系统

使用方法:
1. 设置环境变量: export LAZYLLM_QWEN_API_KEY=your_key
2. 修改 dataset_path 指向你的文档目录
3. 运行: python basic_rag.py
"""

import lazyllm

# 创建文档对象
documents = lazyllm.Document(
    dataset_path="/path/to/your/doc/dir",  # 修改为你的文档目录
    embed=lazyllm.OnlineEmbeddingModule(),
    manager=False
)

# 创建检索器
retriever = lazyllm.Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    similarity_cut_off=0.003,
    topk=3
)

# 创建大模型
llm = lazyllm.OnlineChatModule()

# 设置提示词
prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# 查询和生成
print("RAG 知识库问答系统（输入 'quit' 退出）\n")

while True:
    query = input("query(enter 'quit' to exit): ")
    if query == "quit":
        break

    # 检索相关文档
    doc_node_list = retriever(query=query)

    # 生成回答
    res = llm({
        "query": query,
        "context_str": "".join([node.get_content() for node in doc_node_list]),
    })

    print(f"answer: {res}\n")
