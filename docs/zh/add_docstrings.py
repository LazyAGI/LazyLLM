import sys
sys.path.append('.')
import lazyllm
from lazynote.manager import SimpleManager
manager = SimpleManager(pattern='fill')
manager.traverse(lazyllm, skip_modules=[
    'lazyllm.components.deploy.relay.server',
    'lazyllm.components.deploy.lmdeploy',
    'lazyllm.components.deploy.relay.base',
    'lazyllm.components.finetune.easyllm',
    'lazyllm.tools.rag.component.bm25_retriever',
    'lazyllm.docs'
])
