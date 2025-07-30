import io
import base64
import matplotlib.pyplot as plt
from lazyllm import ChatPrompter, OnlineChatModule
import re
import json

# 图表渲染
def render_chart(config: dict) -> str:
    try:
        chart_type = config.get("series", [{}])[0].get("type", "bar")
        series = config.get("series", [{}])[0]
        fig, ax = plt.subplots()

        if chart_type == "pie":
            data = series.get("data", [])
            labels = [item["name"] for item in data]
            values = [item["value"] for item in data]
            ax.pie(values, labels=labels, autopct="%1.1f%%")
            ax.set_title("饼图")

        elif chart_type in ["bar", "line"]:
            x = config.get("xAxis", {}).get("data", [])
            y = series.get("data", [])
            if chart_type == "bar":
                ax.bar(x, y)
                ax.set_title("柱状图")
            else:
                ax.plot(x, y, marker="o")
                ax.set_title("折线图")
            ax.set_xlabel("X轴")
            ax.set_ylabel("Y轴")

        else:
            return f"⚠️ 暂不支持的图表类型：{chart_type}"

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return f"![chart](data:image/png;base64,{base64_image})"

    except Exception as e:
        return f"⚠️ 图表渲染失败: {e}"


# Markdown 表格生成
def render_table(config: dict) -> str:
    chart_type = config.get("series", [{}])[0].get("type", "bar")

    if chart_type in ["bar", "line"]:
        x = config.get("xAxis", {}).get("data", [])
        y = config.get("series", [{}])[0].get("data", [])
        header = "| X轴 | Y轴 |"
        separator = "|---|---|"
        rows = [f"| {x[i]} | {y[i]} |" for i in range(len(x))]
        return "\n".join([header, separator] + rows)

    elif chart_type == "pie":
        data = config.get("series", [{}])[0].get("data", [])
        header = "| 类别 | 数值 |"
        separator = "|---|---|"
        rows = [f"| {item['name']} | {item['value']} |" for item in data]
        return "\n".join([header, separator] + rows)

    return "⚠️ 不支持的图表类型"


# 构建图表生成 pipeline
def create_chart_gen_tool(prompt: str, source: str):
    prompter = ChatPrompter(prompt)
    llm = OnlineChatModule(source=source).prompt(prompter)

    def pipeline(input_text: str):
        config = llm(input_text)

        try:
            if isinstance(config, str):
                # 清除 markdown 代码块标记和解释文字
                config = config.strip()
                config = re.sub(r"^```(json)?", "", config)
                config = re.sub(r"```$", "", config)
                config = config.strip()
                config = json.loads(config)

            chart = render_chart(config)
            table = render_table(config)
            return f"{chart}\n\n### 📋 数据表格：\n\n{table}"
        except Exception as e:
            return f"⚠️ 图表生成失败: {e}"

    return pipeline
