import sys
import argparse
import os

sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager.llm_manager import LLMDocstringManager
import lazyllm
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='自动处理项目中的Python包和模块的文档字符串')
    parser.add_argument('--path', type=str, default='.', help='项目根目录路径')
    args = parser.parse_args()
    
    process_directory(args.path)