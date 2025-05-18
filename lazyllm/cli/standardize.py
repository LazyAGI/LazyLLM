import os
import sys

sys.path.append('.')
sys.path.append('./docs/scripts')


def standardize(commands):
    if not commands or len(commands) < 3:
        print("Usage: lazyllm standardize <model> <language> <project_path> ")
        sys.exit(1)

    model = commands[0]
    language = commands[1]
    project_path = commands[2]
    if not os.path.exists(project_path):
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)

    support_model = ["qwen", "deepseek", "gpt"]
    if model not in support_model:
        print(f"Please select from the supported models: {support_model}")
        sys.exit(1)

    try:
        from lazynote.agent.git_agent import GitAgent
        from lazyllm import OnlineChatModule
        llm = OnlineChatModule(source=model, stream=False)
        agent = GitAgent(project_path=project_path, llm=llm, language=language)
        agent.standardize_project()

    except Exception as e:
        print(f"Error during project standardization: {e}")
        sys.exit(1)
