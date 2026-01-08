import os
import tempfile
from lazyllm.tools.rag.transform import RecursiveSplitter
from lazyllm import Document, Retriever, LOG


def run():
    documents = Document(url='http://127.0.0.1:9977', name='doc_example')
    retriever = Retriever(doc=documents, group_name='sentences', topk=3)
    res = retriever('What is the meaning of life?')
    LOG.info(res)
run()