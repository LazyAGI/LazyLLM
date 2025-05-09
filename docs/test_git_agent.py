import sys
import argparse
import os

sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager.llm_manager import LLMDocstringManager
import lazyllm
import importlib
import time

from lazynote.agent.git_agent import GitAgent

agent = GitAgent(project_path="/home/mnt/jisiyuan/projects/tmp/myproject")
agent.standardize_project()
time.sleep(100)
