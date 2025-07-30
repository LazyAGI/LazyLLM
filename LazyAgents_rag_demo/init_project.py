import os

structure = {
    "data": ["sales_data.csv"],
    "db": ["sales.db"],
    "docs": ["faq.md"],
    "tools": ["query_executor.py", "chart_plotter.py", "rag_helper.py", "summarizer.py"],
    "agents": ["rag_agent.py", "sql_agent.py", "plot_agent.py", "summary_agent.py", "pipeline.py"],
    "frontend": ["app.py"],
    "": ["main.py", "config.py", "requirements.txt", ".env"]
}

for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        with open(os.path.join(folder, f), "w") as fp:
            if f.endswith(".py"):
                fp.write(f"# {f}\n")
            elif f == "requirements.txt":
                fp.write("lazyllm\nopenai\ndashscope\nmatplotlib\npandas\nchromadb\nsentence-transformers\nstreamlit\n")
            elif f == "sales_data.csv":
                fp.write("date,region,sales\n2024-01,华东,12000\n")
            elif f == ".env":
                fp.write("LAZYLLM_QWEN_API_KEY=你的API密钥\n")
