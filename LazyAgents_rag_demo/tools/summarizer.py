# tools/summarizer.py

from lazyllm import OnlineChatModule

# 可以在 pipeline 中提前实例化
summarizer = OnlineChatModule(
    source="qwen",
    model="qwen-plus",
    stream=False,
    prompt=(
        "你是一个顶尖的数据分析专家，请根据以下数据和用户问题，总结出一句简洁的结论，控制在50字以内。"
    )
)

def generate_summary(question: str, df):
    data_str = df.to_csv(index=False)
    full_prompt = f"用户问题：{question}\n数据内容：\n{data_str}"
    return summarizer.forward(full_prompt)
# summarizer.py
