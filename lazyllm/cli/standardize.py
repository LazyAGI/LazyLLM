import os
import sys
from docs.scripts.lazynote.agent.git_agent import GitAgent
from lazyllm import OnlineChatModule

def standardize(commands):
    """
    标准化项目的入口函数
    
    Args:
        commands: 命令行参数列表，第一个参数为项目路径，第二个参数为可选的模型名称
    """
    if not commands or len(commands) < 1:
        print("Usage: lazyllm standardize <model> <project_path> ")
        sys.exit(1)
    
    model = commands[0]
    project_path = commands[1]
    if not os.path.exists(project_path):
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)
    
    support_model = ["qwen", "deepseek", "gpt"]
    if not model in support_model:
        print(f"Please select from the supported models: {support_model}")
        sys.exit(1)
        
    try:
        llm = OnlineChatModule(source=model, stream=False)
        agent = GitAgent(project_path=project_path, llm=llm)
        agent.standardize_project()
        
    except Exception as e:
        print(f"Error during project standardization: {e}")
        sys.exit(1)