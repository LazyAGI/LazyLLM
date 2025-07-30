# planner_agent.py
import sys
sys.path.append("/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo")  # 根据你的项目路径调整

from tools.data_plot_tool import create_chart_gen_tool
from tools.text2sql_tool import create_text2sql_pipeline
from tools.rag_tool import create_rag_tool
from lazyllm import OnlineChatModule, ChatPrompter, WebModule
import re
import textwrap
import pandas as pd

# 初始化各个 pipeline
rag_pipeline = create_rag_tool(
    prompt="你是一个知识问答助手，请结合上下文正确回答用户的问题。",
    dataset_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs",
    source="qwen",
    stream=True
)

text2sql_pipeline = create_text2sql_pipeline(
    source="qwen",
    db_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs/blackfriday.db"
)

chart_pipeline = create_chart_gen_tool(
    prompt="""你是一个图表 JSON 配置生成器。请根据用户输入的需求，仅输出符合以下结构的 JSON 配置：
- bar/line 图：xAxis.data 是横坐标，series.data 是纵坐标值。
- pie 图：series.data 是包含 name 和 value 的对象数组。
请严格输出纯 JSON，无任何 markdown、解释或注释。例如：
{
  "xAxis": { "data": ["A", "B", "C"] },
  "series": [{ "type": "bar", "data": [30, 50, 20] }]
}
""",
    source="qwen"
)

# 调度 LLM：让它输出实际可执行的 Python 代码
PLANNER_PROMPT = """
你是一个数据分析调度助手。
你可以调用以下函数：
- rag_pipeline(question: str) → 返回知识性答案
- text2sql_pipeline(question: str) → 执行 SQL 查询并返回结果
- chart_pipeline(text: str) → 根据查询结果生成图表

用户会给出一个自然语言问题，请你输出对应的函数调用代码，并将最终答案赋值到变量 result。如果有多个步骤，请使用 Python 函数调用的方式进行调度。如果使用text2sql_pipeline问题没有找到答案或执行失败时，要切换使用rag_pipeline继续寻找答案。
当用户有绘图需求时，要记得调用 chart_pipeline 生成图表。
请确保 result 变量包含最终的答案或图表 JSON 配置。

注意：
-请只输出纯 Python 可执行代码，不要添加 markdown 代码块（如 ```python ... ```），也不要输出解释。
- 不要使用 import 或 from 导入任何模块。
- 所有可用函数已经注入到当前作用域（包括：rag_pipeline、text2sql_pipeline、chart_pipeline）。
- 请直接调用这些函数。
- 不要自己写 SQL 语句。
- 如果需要生成图表，请调用 chart_pipeline。
- 最终必须设置变量 result = ...
""".strip()

planner_llm = OnlineChatModule(source="openai",model="Pro/moonshotai/Kimi-K2-Instruct", base_url="https://api.siliconflow.cn/v1", api_key="sk-wkuipstnxqfqdqrimazhnwvtxftauxhrbtshxidjhwrccqvh").prompt(ChatPrompter(PLANNER_PROMPT))


def extract_python_code(text):
    """提取最上面的 Python 函数调用块（包含 result=）"""
    # 删除 markdown ```包裹
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())

    # 移除解释性段落（比如 SQL:、解释:、LLM回复:）
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        if re.match(r"^\s*(SQL|解释|LLM|实际执行结果|结果)：?", line):
            break  # 遇到解释类内容就停止
        code_lines.append(line)

    return textwrap.dedent("\n".join(code_lines)).strip()


def markdown_to_dataframe(markdown_table: str) -> pd.DataFrame:
    try:
        lines = markdown_table.strip().splitlines()
        if len(lines) < 3:
            return None
        header = [h.strip() for h in lines[0].split("|") if h.strip()]
        rows = []
        for line in lines[2:]:  # Skip header and separator
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                rows.append(cells)
        return pd.DataFrame(rows, columns=header)
    except Exception:
        return None



def planner_pipeline(user_input: str):
    try:
        raw_output = planner_llm(user_input)
        print("🧠 LLM 原始输出：\n", raw_output)

        code = extract_python_code(raw_output)
        print("✅ 清理后的代码：\n", code)

        # 检查必须有 result=
        if "result" not in code:
            return "⚠️ 未检测到 result = ... 赋值，请确认大模型是否正确生成代码"

        # 构造执行上下文
        local_env = {
            "user_input": user_input,
            "rag_pipeline": rag_pipeline,
            "text2sql_pipeline": text2sql_pipeline,
            "chart_pipeline": chart_pipeline,
        }

        # 执行代码
        exec(code, {}, local_env)
        result = local_env.get("result", "")

        # 清理可能返回的 markdown 格式
        if isinstance(result, str) and result.strip().startswith("```"):
            result = re.sub(r"^```[a-zA-Z]*\n?", "", result.strip())
            result = re.sub(r"\n?```$", "", result.strip())

        return result if result else "⚠️ 执行完成，但返回为空"

    except Exception as e:
        return f"❌ 执行失败：{e}"

# 启动 Web 服务
if __name__ == "__main__":
    WebModule(planner_pipeline, port=range(23491, 23500)).start().wait()
