import lazyllm
import time
from unittest.mock import patch
from lazyllm.tools.rag.document import Document
from lazyllm.tools.rag.graph_document import GraphDocument
from lazyllm.tools.rag.graph_retriever import GraphRetriever


doc = Document(dataset_path='/mnt/lustre/share_data/zhangyc/afs_space/work/test_graphrag', name='test_graphrag')

# Mock kb_files 的值
mock_kb_files = [
    '/mnt/lustre/share_data/zhangyc/afs_space/work/test_graphrag/shuihu_parts.txt',
]

# 使用 patch 来 mock _list_all_files_in_kb 方法
with patch.object(doc, '_list_all_files_in_kb', return_value=mock_kb_files):
    graph_document = GraphDocument(doc, 'default')
    graph_document.start()
    user_input = input('Press Enter to start graphrag index: ')

    graph_document.start_graphrag_index(override=False)

    lazyllm.LOG.info('start_graphrag_index done')
    time.sleep(600)

    graph_retriever = GraphRetriever(graph_document, 'default')

    print(graph_retriever.forward('鲁智深为什么要打镇关西'))
