import sys
import argparse

sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager import SimpleManager
import lazyllm

parser = argparse.ArgumentParser()
parser.add_argument('--replace', action='store_true', help='Execute the replace part of the code.')
parser.add_argument('--clean', action='store_true', help='clean code docs.')
args = parser.parse_args()

skip_list = [
    'lazyllm.components.deploy.relay.server',
    'lazyllm.components.deploy.relay.base',
    'lazyllm.components.finetune.easyllm',
    'lazyllm.tools.rag.component.bm25_retriever',
]

if args.replace or args.clean:
    manager = SimpleManager(pattern='clear', skip_on_error=True)
    manager.traverse(lazyllm, skip_modules=skip_list)

if not args.clean:
    manager = SimpleManager(pattern='fill', skip_on_error=True)
    manager.traverse(lazyllm, skip_modules=skip_list)
