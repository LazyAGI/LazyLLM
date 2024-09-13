import os
import lazyllm
from lazyllm import Document
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import SimpleDirectoryReader, DocNode

class YmlReader(ReaderBase):
    def _load_data(self, file, extra_info=None, fs=None):
        with open(file, 'r') as f:
            data = f.read()
        return [DocNode(text=data, metadata=extra_info or {})]

class TestRagReader(object):
    @classmethod
    def setup_class(cls):
        cls.doc = Document(dataset_path="ci_data", create_ui=False)
        cls.datasets = os.path.join(lazyllm.config['data_path'], "ci_data/default/__data/sources")

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

    def test_register_reader(self):
        self.doc.add_reader("/**/*.yml", YmlReader)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc._impl._impl.directory_reader.load_data(input_files=files)
        assert len(docs) == 1
