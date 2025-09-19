# flake8: noqa E501
from lazyllm.prompts import PromptTemplate, FewShotPromptTemplate

# Below is the default QA transform prompt
DEFAULT_QA_EGS_TEMPLATE = PromptTemplate.from_template("""
egample {index}:
input:
{input}
output:
{output}
""")

DEFAULT_QA_EXAMPLES = [
    {
        "index": 1,
        "input": "Hello, I am an AI robot developed by SenseTime, named LazyLLM.\nMy mission is to assist you in building the most powerful large-scale model applications with minimal cost.",
        "output": """Q: What is the name of the AI robot developed by SenseTime?
A: LazyLLM.
Q: Which company developed the AI robot named LazyLLM?
A: SenseTime.
Q: What is the mission of the AI robot named LazyLLM?
A: To assist in building the most powerful large-scale model applications with minimal cost.""",
    },
    {
        "index": 2,
        "input": "你好，我是由商汤开发的人工智能机器人，我叫LazyLLM。我的使命是协助您，用最低的成本，构建最强大的大模型应用。",
        "output": """Q: 商汤科技开发的AI机器人叫什么名字？
A: LazyLLM。
Q: 名为LazyLLM的AI机器人是由哪家公司开发的？
A: 商汤科技。
Q: 名为LazyLLM的AI机器人的使命是什么？
A: 以最小成本协助构建最强大的大规模模型应用程序。""",}]

DEFAULT_QA_PREFIX = """## Role: Q&A Generation Assistant.  
You are a Q&A generation assistant. Your task is to generate {num_pairs_min}-{num_pairs_max} question-answer pairs based on the context.  
## Constraints:
- Each question must be answerable directly and specifically from the text.  
- The answers should be concise and accurate.  
- Do not include information not supported by the context.  
- Prefer factual, detail-rich questions over vague ones.  
- Questions must be self-contained: avoid ambiguous references such as "this text", "this passage", "here", or "the above". Each question should stand alone without requiring external context. 
## Text Format:
The input is a string contains the user's raw input text
## Output Format 
Format your output as:
Q: <question>  
A: <answer>  
Q: <question>  
A: <answer> 
## Example:"""

DEFAULT_QA_SUFFIX = """You should not have any unnecessary output. Lets begin:"""

# 使用 FewShotPromptTemplate 重构 TRANSFORM_QA_PROMPT
DEFAULT_QA_PROMPT = FewShotPromptTemplate(
    prefix=DEFAULT_QA_PREFIX,
    suffix=DEFAULT_QA_SUFFIX,
    examples=DEFAULT_QA_EXAMPLES,
    egs_prompt_template=DEFAULT_QA_EGS_TEMPLATE,
    required_vars=["num_pairs_min", "num_pairs_max"],
    partial_vars={}
)