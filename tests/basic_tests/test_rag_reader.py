import os
import lazyllm
import pytest
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import SimpleDirectoryReader, DocNode, Document
from lazyllm.tools.rag.dataReader import RAG_DOC_CREATION_DATE

class YmlReader(ReaderBase):
    def _load_data(self, file, fs=None):
        with open(file, 'r') as f:
            data = f.read()
            node = DocNode(text=data)
            node._content = "Call the class YmlReader."
            return [node]

def processYml(file):
    with open(file, 'r') as f:
        data = f.read()
        node = DocNode(text=data)
        node._content = "Call the function processYml."
        return [node]

def processYmlWithMetadata(file):
    with open(file, 'r') as f:
        data = f.read()
        node = DocNode(text=data, metadata=dict(m='m'), global_metadata={RAG_DOC_CREATION_DATE: '00-00'})
        node._content = 'Call the function processYml.'
        return [node]

class TestRagReader(object):
    def setup_method(self):
        self.doc1 = Document(dataset_path="ci_data/rag_reader_full", manager=False)
        self.doc2 = Document(dataset_path="ci_data/rag_reader_full", manager=False)
        self.datasets = os.path.join(lazyllm.config['data_path'], "ci_data/rag_reader_full")

    def teardown_method(self):
        self.doc1._impl._local_file_reader = {}
        self.doc2._impl._local_file_reader = {}
        type(self.doc1._impl)._registered_file_reader = {}

    def test_reader_file(self):
        files = [os.path.join(self.datasets, "联网搜索.pdf"), os.path.join(self.datasets, "说明文档测试.docx")]
        reader = SimpleDirectoryReader(input_files=files)
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 3

    # TODO: remove *.pptx and *.jpg, *.png in mac and win
    @pytest.mark.skip_on_mac
    @pytest.mark.skip_on_win
    def test_reader_dir(self):
        input_dir = self.datasets
        reader = SimpleDirectoryReader(input_dir=input_dir,
                                       exclude=["*.yml", "*.pdf", "*.docx", "*.mp4"])
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 23

    def test_register_local_reader(self):
        self.doc1.add_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == "Call the function processYml."

    def test_register_global_reader(self):
        Document.register_global_reader("**/*.yml", processYml)
        files = [os.path.join(self.datasets, "reader_test.yml")]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == "Call the function processYml."

    def test_register_reader_metadata(self):
        self.doc1.add_reader('**/*.yml', processYmlWithMetadata)
        files = [os.path.join(self.datasets, 'reader_test.yml')]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == 'Call the function processYml.'
        assert docs[0].metadata.get('m') == 'm'
        assert docs[0].global_metadata.get(RAG_DOC_CREATION_DATE) == '00-00'

    def test_register_local_and_global_reader(self):
        files = [os.path.join(self.datasets, "reader_test.yml")]

        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        assert docs1[0].text != "Call the class YmlReader." and docs1[0].text != "Call the function processYml."
        Document.add_reader("**/*.yml", processYml)
        self.doc1.add_reader("**/*.yml", YmlReader)
        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        docs2 = self.doc2._impl._reader.load_data(input_files=files)
        assert docs1[0].text == "Call the class YmlReader." and docs2[0].text == "Call the function processYml."
