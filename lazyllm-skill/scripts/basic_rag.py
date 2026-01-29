# Part0
# 导入了 lazyllm
import lazyllm

# Part1
# 从本地加载知识库目录，并使用内置的 OnlineEmbeddingModule 作为向量模型。
documents = lazyllm.Document(dataset_path='/path/to/your/doc/dir',
                             embed=lazyllm.OnlineEmbeddingModule(),
                             manager=False)

# Part2
# 创建一个用于检索文档的 Retriever
# 并使用内置的 CoarseChunk（参考 [CoarseChunk 的定义][llm.tools.Retriever]）将文档按指定的大小分块
# 然后使用内置的 bm25_chinese 作为相似度计算函数，并且丢弃相似度小于 0.003 的结果
# 最后取最相近的 3 篇文档。
retriever = lazyllm.Retriever(doc=documents,
                              group_name='CoarseChunk',
                              similarity='bm25_chinese',
                              similarity_cut_off=0.003,
                              topk=3)

# Part3
# 创建用来回答问题的大模型实例。
llm = lazyllm.OnlineChatModule()

# Part4
# 由于需要大模型基于我们提供的文档回答问题，我们在提问的时候需要告诉大模型哪些是参考资料，哪个是我们的问题。
# 这里使用内置的 ChatPrompter 将 Retriever 返回的文档内容作为参考资料告诉大模型。
# 这里用到的 ChatPrompter 两个参数含义如下：
# 1. instruction：提供给大模型的指引内容；
# 2. extra_keys：从传入的 dict 中的哪个字段获取参考资料。
prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

# Part5
# 打印提示信息，等待用户输入要查询的内容。
query = input("query(enter 'quit' to exit): ")
if query == "quit":
    exit(0)

# Part6
# 主流程：接收用户的输入，使用 Retriever 根据用户输入的 query 检索出相关的文档，
# 然后把 query 和参考资料 context_str 打包成一个 dict 传给大模型，并等待结果返回。
doc_node_list = retriever(query=query)
res = llm({
    'query': query,
    'context_str': ''.join([node.get_content() for node in doc_node_list]),
})

# Part7
# 结果打印到屏幕上。
print(f'answer: {res}')
