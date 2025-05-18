import sys
import os
import argparse

sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager.custom import CustomManager
import lazyllm
from lazyllm import OnlineChatModule

parser = argparse.ArgumentParser()
parser.add_argument('--replace', action='store_true', help='Execute the replace part of the code.')
parser.add_argument('--clean', action='store_true', help='clean code docs.')
args = parser.parse_args()

skip_list = [
    'lazyllm.components.deploy.relay.server',
    'lazyllm.components.deploy.relay.base',
    'lazyllm.components.finetune.easyllm',
    'lazyllm.tools.rag.component.bm25_retriever',
    'lazyllm.cli'
]

if args.replace or args.clean:
    manager = CustomManager(pattern='clear', skip_on_error=True)
    manager.traverse(lazyllm, skip_modules=skip_list)

if not args.clean:
    language = os.getenv('LAZYLLM_LANGUAGE', 'ENGLISH')
    language = 'en' if language == 'ENGLISH' else 'zh'
    manager = CustomManager(llm=OnlineChatModule(source='deepseek', stream=False),
                            language=language, pattern='fill', skip_on_error=True)
    manager.traverse(lazyllm, skip_modules=skip_list)