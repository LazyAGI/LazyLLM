import lazyllm
from lazyllm import pipeline, Document
from lazyllm.tools.rag.doc_impl import RetrieverV2
from lazyllm.tools.rag.rerank import register_reranker
from lazyllm.tools.rag.index import register_similarity

@register_similarity(mode="text")
def dummy2(query: str, node, **kwargs):
    return len(node.text)

@register_reranker
def dummy_reranker(node, **kwargs):
    if "我们" in node.text:
        return node
    

prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
# embed_model = lazyllm.TrainableModule('bge-large-zh-v1.5')
embed_model = None
documents = Document(dataset_path='/home/mnt/yewentao/data/develop_data', 
                     embed=embed_model, create_ui=True)
documents._impl._impl.create_node_group(name='block', transform=lambda t: t.split('。'))
documents._impl._impl.create_node_group(name='doc-summary', transform=lambda t: t[:10])
documents._impl._impl.create_node_group(name='block-summary', transform=lambda t: t[:8], parent='block')
documents._impl._impl.create_node_group(name='sentence', transform=lambda t: t.split('，'), parent='block')
documents._impl._impl.create_node_group(name='block-label', transform=lambda t: t[:3], parent='block')
documents._impl._impl.create_node_group(name='sentence-label', transform=lambda t: t[-2:], parent='sentence')


with pipeline() as ppl:
    ppl.retriever1 = RetrieverV2(documents, group_name='block-label', similarity='bm25_chinese', similarity_cut_off=0.003, topk=100)
    ppl.find_parent = documents._impl._impl.find_parent(group='block')
    ppl.find_children = documents._impl._impl.find_children(group='sentence-label')
    # In this general test, we don't use llm, just print the nodes
    ppl.to_str = lambda nodes: str([str(node) for node in nodes])
    
mweb = lazyllm.WebModule(ppl, port=23456).start().wait()