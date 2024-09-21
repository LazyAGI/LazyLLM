import os
import lazyllm
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import SimpleDirectoryReader, DocNode, Document

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
        node = DocNode(text=data, metadata=extra_info or {})
        node.text = "Call the function processYml."
        return [node]

class TestRagReader(object):
    def setup_method(self):
        self.doc1 = Document(dataset_path="ci_data/rag_reader", manager=False)
        self.doc2 = Document(dataset_path="ci_data/rag_reader", manager=False)
        self.datasets = os.path.join(lazyllm.config['data_path'], "ci_data/rag_reader/default/__data/sources")

    def teardown_method(self):
        self.doc1._local_file_reader = {}
        self.doc2._local_file_reader = {}
        Document._registered_file_reader = {}

    def test_reader_file(self):
        files = [os.path.join(self.datasets, "联网搜索.pdf"), os.path.join(self.datasets, "说明文档测试.docx")]
        reader = SimpleDirectoryReader(input_files=files)
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 3

    def test_reader_dir(self):
        input_dir = self.datasets
        reader = SimpleDirectoryReader(input_dir=input_dir,
                                       exclude=["*.jpg", "*.mp3", "*.yml", "*.pdf", ".docx", "*.pptx"])
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 13

    def test_register_local_reader(self):
        self.doc1.add_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc1._impl._impl.directory_reader.load_data(input_files=files)
        assert docs[0].text == "Call the function processYml."

    def test_register_global_reader(self):
        Document.register_global_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc1._impl._impl.directory_reader.load_data(input_files=files)
        assert docs[0].text == "Call the function processYml."

    def test_register_local_and_global_reader(self):
        files = [os.path.join(self.datasets, "reader_test.yml")]

        docs1 = self.doc1._impl._impl.directory_reader.load_data(input_files=files)
        assert docs1[0].text != "Call the class YmlReader." and docs1[0].text != "Call the function processYml."
        Document.add_reader("**/*.yml", processYml)
        self.doc1.add_reader("**/*.yml", YmlReader)
        docs1 = self.doc1._impl._impl.directory_reader.load_data(input_files=files)
        docs2 = self.doc2._impl._impl.directory_reader.load_data(input_files=files)
        assert docs1[0].text == "Call the class YmlReader." and docs2[0].text == "Call the function processYml."
