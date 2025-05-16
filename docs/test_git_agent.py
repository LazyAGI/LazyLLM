import sys
import argparse
import os

sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager.llm_manager import LLMDocstringManager
import lazyllm
from lazyllm import OnlineChatModule
import importlib
import time

from lazynote.agent.git_agent import GitAgent

agent = GitAgent(project_path="/home/mnt/jisiyuan/projects/tmp/magic-html/", llm=OnlineChatModule(source='deepseek', stream=False))
agent.standardize_project()
agent.start_mkdocs_server()
