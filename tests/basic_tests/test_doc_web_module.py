import lazyllm
from lazyllm.tools.rag import Document
import time


def test_web_doc_module():
    kb_path = "/home/mnt/jisiyuan/projects/LazyLLM/data_kv"
    docs = Document(kb_path, server=True, manager=True)
    docs.create_kb_group(name="test_group")
    docs.start()
    time.sleep(1000)


if __name__ == "__main__":
    test_web_doc_module()
    