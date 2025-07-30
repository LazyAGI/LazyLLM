# rag_agent.py
import sys
sys.path.append("/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo")  # 必须添加路径
from tools.rag_tool import create_rag_tool
import lazyllm

if __name__ == "__main__":
    prompt = "你是一个知识问答助手，请结合上下文正确回答用户的问题。"

    rag_pipeline = create_rag_tool(
        prompt=prompt,
        source="qwen",
        stream=True,
        dataset_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs"
    )

    lazyllm.WebModule(rag_pipeline, port=range(23471, 23480)).start().wait()
