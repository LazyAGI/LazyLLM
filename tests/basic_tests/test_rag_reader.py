import os
import lazyllm
from lazyllm import Document
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import SimpleDirectoryReader, DocNode

class YmlReader(ReaderBase):
    def _load_data(self, file, extra_info=None, fs=None):
        with open(file, 'r') as f:
            data = f.read()
            node = DocNode(text=data, metadata=extra_info or {})
            node.text = "Call the class YmlReader."
        return [node]

def processYml(file, extra_info=None):
    with open(file, 'r') as f:
        data = f.read()
    return [DocNode(text=data, metadata=extra_info or {})]

class TestRagReader(object):
    @classmethod
    def setup_class(cls):
        cls.doc = Document(dataset_path="ci_data/rag_reader", create_ui=False)
        cls.datasets = os.path.join(lazyllm.config['data_path'], "ci_data/rag_reader/default/__data/sources")

    def test_reader_file(self):
        files = [os.path.join(self.datasets, "联网搜索.pdf"), os.path.join(self.datasets, "说明文档测试.docx")]
        reader = SimpleDirectoryReader(input_files=files)
        docs = []
        for doc in reader():
            print(doc)
            docs.append(doc)
        print(len(docs))
        assert len(docs) == 3

    def test_reader_dir(self):
        input_dir = self.datasets
        reader = SimpleDirectoryReader(input_dir=input_dir,
                                       exclude=["*.jpg", "*.mp3", "*.yml", "*.pdf", ".docx", "*.pptx"])
        docs = []
        for doc in reader():
            print(doc)
            docs.append(doc)
        print(len(docs))
        assert len(docs) == 13

    def test_register_local_reader(self):
        self.doc.add_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc._impl._impl.directory_reader.load_data(input_files=files)
        assert len(docs) == 1

    def test_register_global_reader(self):
        Document.register_global_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc._impl._impl.directory_reader.load_data(input_files=files)
        assert len(docs) == 1

    def test_register_local_and_global_reader(self):
        Document.register_global_reader("**/*.yml", processYml)
        self.doc.add_reader("**/*.yml", YmlReader)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc._impl._impl.directory_reader.load_data(input_files=files)
        assert docs[0].text == "Call the class YmlReader."
