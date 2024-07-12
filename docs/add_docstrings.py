import sys
sys.path.append('.')
sys.path.append('./docs/scripts')
from lazynote.manager import SimpleManager
import lazyllm
manager = SimpleManager(pattern='clear', skip_on_error=True)
manager.traverse(lazyllm, skip_modules=[
    'lazyllm.components.deploy.relay.server',
    'lazyllm.components.deploy.lmdeploy',
    'lazyllm.components.deploy.relay.base',
    'lazyllm.components.finetune.easyllm',
    'lazyllm.tools.rag.component.bm25_retriever',
    'lazyllm.docs'
])
