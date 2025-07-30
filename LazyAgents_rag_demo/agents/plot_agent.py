# chart_generator_agent.py
import sys
sys.path.append("/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo")  # 修改为你的路径

from tools.data_plot_tool import create_chart_gen_tool
import lazyllm

if __name__ == "__main__":
    prompt = (
        """你是一个图表 JSON 配置生成器。请根据用户输入的需求，仅输出符合以下结构的 JSON 配置：
- bar/line 图：xAxis.data 是横坐标，series.data 是纵坐标值。
- pie 图：series.data 是包含 name 和 value 的对象数组。

请严格输出纯 JSON，无任何 markdown、解释或注释。例如：
{
  "xAxis": { "data": ["A", "B", "C"] },
  "series": [{ "type": "bar", "data": [30, 50, 20] }]
}
"""
    )

    chart_pipeline = create_chart_gen_tool(
        prompt=prompt,
        source="qwen"  
    )

    #lazyllm.WebModule(chart_pipeline, port=range(23481, 23490)).start().wait()
