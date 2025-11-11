import lazyllm
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
with patch.object(doc, '_list_all_files_in_dataset', return_value=mock_kb_files):
    graph_document = GraphDocument(doc)
    graph_document.start()
    user_input = input('Press Enter when files are ready')

    graph_document.init_graphrag_kg()
    # Now you need to edit $dataset_path/.graphrag_kg/settings.yaml

    user_input = input('Press Enter when settings.yaml is ready')
    graph_document.start_graphrag_index(override=False)

    status_dict = graph_document.graphrag_index_status()
    lazyllm.LOG.info(f'graphrag index status: {status_dict}')

    lazyllm.LOG.info('start_graphrag_index done')
    user_input = input('Press Enter to start graphrag retriever: ')

    graph_retriever = GraphRetriever(graph_document, 'default')

    print(graph_retriever.forward('鲁智深为什么要打镇关西'))
