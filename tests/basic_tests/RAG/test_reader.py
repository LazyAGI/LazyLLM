import os
import lazyllm
import tiktoken
from lazyllm.tools.rag.transform import SentenceSplitter
import pytest
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag.readers.readerBase import TxtReader
from lazyllm.tools.rag import SimpleDirectoryReader, DocNode, Document
from lazyllm.tools.rag.dataReader import RAG_DOC_CREATION_DATE
import tempfile
import shutil

class YmlReader(ReaderBase):
    def _load_data(self, file, fs=None):
        with open(file, 'r') as f:
            data = f.read()
            node = DocNode(text=data)
            node._content = 'Call the class YmlReader.'
            return [node]

def processYml(file):
    with open(file, 'r') as f:
        data = f.read()
        node = DocNode(text=data)
        node._content = 'Call the function processYml.'
        return [node]

def processYmlWithMetadata(file):
    with open(file, 'r') as f:
        data = f.read()
        node = DocNode(text=data, metadata=dict(m='m'), global_metadata={RAG_DOC_CREATION_DATE: '00-00'})
        node._content = 'Call the function processYml.'
        return [node]

class TestRagReader(object):
    def setup_method(self):
        self.doc1 = Document(dataset_path='ci_data/rag_reader_full', manager=False)
        self.doc2 = Document(dataset_path='ci_data/rag_reader_full', manager=False)
        self.datasets = os.path.join(lazyllm.config['data_path'], 'ci_data/rag_reader_full')

    def teardown_method(self):
        self.doc1._impl._local_file_reader = {}
        self.doc2._impl._local_file_reader = {}
        type(self.doc1._impl)._registered_file_reader = {}

    def test_reader_file(self):
        files = [os.path.join(self.datasets, '联网搜索.pdf'), os.path.join(self.datasets, '说明文档测试.docx')]
        reader = SimpleDirectoryReader(input_files=files)
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 7

    # TODO: remove *.pptx and *.jpg, *.png in mac and win
    @pytest.mark.skip_on_mac
    @pytest.mark.skip_on_win
    def test_reader_dir(self):
        input_dir = self.datasets
        reader = SimpleDirectoryReader(input_dir=input_dir,
                                       exclude=['*.yml', '*.pdf', '*.docx', '*.mp4'])
        docs = []
        for doc in reader():
            docs.append(doc)
        assert len(docs) == 23

    def test_register_local_reader(self):
        self.doc1.add_reader('**/*.yml', processYml)
        files = [os.path.join(self.datasets, 'reader_test.yml')]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == 'Call the function processYml.'

    def test_register_global_reader(self):
        Document.register_global_reader('**/*.yml', processYml)
        files = [os.path.join(self.datasets, 'reader_test.yml')]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == 'Call the function processYml.'

    def test_register_reader_metadata(self):
        self.doc1.add_reader('**/*.yml', processYmlWithMetadata)
        files = [os.path.join(self.datasets, 'reader_test.yml')]
        docs = self.doc1._impl._reader.load_data(input_files=files)
        assert docs[0].text == 'Call the function processYml.'
        assert docs[0].metadata.get('m') == 'm'
        assert docs[0].global_metadata.get(RAG_DOC_CREATION_DATE) == '00-00'

    def test_register_local_and_global_reader(self):
        files = [os.path.join(self.datasets, 'reader_test.yml')]

        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        assert docs1[0].text != 'Call the class YmlReader.' and docs1[0].text != 'Call the function processYml.'
        Document.add_reader('**/*.yml', processYml)
        self.doc1.add_reader('**/*.yml', YmlReader)
        docs1 = self.doc1._impl._reader.load_data(input_files=files)
        docs2 = self.doc2._impl._reader.load_data(input_files=files)
        assert docs1[0].text == 'Call the class YmlReader.' and docs2[0].text == 'Call the function processYml.'

    def test_register_post_action_for_default_reader(self):
        def action(x):
            x += 'here in action'
            return DocNode(text=x)

        lazyllm.tools.rag.add_post_action_for_default_reader('*.md', action)
        files = [os.path.join(self.datasets, 'README.md')]
        r = self.doc1._impl._reader.load_data(input_files=files)
        assert len(r) > 0 and 'here in action' in r[0].text

    def test_register_post_action_for_default_reader_docnode(self):
        def action(x):
            return DocNode(text=(x.text + 'here in action'))

        lazyllm.tools.rag.add_post_action_for_default_reader('*.md', action)
        files = [os.path.join(self.datasets, 'README.md')]
        r = self.doc1._impl._reader.load_data(input_files=files)
        assert len(r) > 0 and 'here in action' in r[0].text

    def test_register_post_action_for_default_reader_transform(self):
        lazyllm.tools.rag.add_post_action_for_default_reader('*.md', SentenceSplitter(128, 16))
        files = [os.path.join(self.datasets, 'README.md')]
        r = self.doc1._impl._reader.load_data(input_files=files)
        tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        assert len(r) > 1 and len(tiktoken_tokenizer.encode(r[0].text, allowed_special='all')) < 128

    def test_mixed_encoding_files(self):
        temp_dir = tempfile.mkdtemp()
        try:
            utf8_file = os.path.join(temp_dir, 'utf8_test.txt')
            with open(utf8_file, 'w', encoding='utf-8') as f:
                f.write('这是UTF-8编码的中文内容\nThis is UTF-8 content')

            gbk_file = os.path.join(temp_dir, 'gbk_test.txt')
            with open(gbk_file, 'w', encoding='gbk') as f:
                f.write('这是GBK编码的中文内容\nThis is GBK content')

            gb2312_file = os.path.join(temp_dir, 'gb2312_test.txt')
            with open(gb2312_file, 'w', encoding='gb2312') as f:
                f.write('这是GB2312编码的中文内容\nThis is GB2312 content')

            latin1_file = os.path.join(temp_dir, 'latin1_test.txt')
            with open(latin1_file, 'w', encoding='latin-1') as f:
                f.write('This is Latin-1 content with special chars: àéîöü')

            reader = SimpleDirectoryReader(input_dir=temp_dir)
            docs = []
            for doc in reader():
                docs.append(doc)

            assert len(docs) == 4, f'Expected 4 docs, got {len(docs)}'

            texts = [doc.text for doc in docs]
            print(texts)
            assert any('UTF-8' in text for text in texts), 'UTF-8 file not read correctly'
            assert any('GBK' in text for text in texts), 'GBK file not read correctly'
            assert any('GB2312' in text for text in texts), 'GB2312 file not read correctly'
            assert any('Latin-1' in text for text in texts), 'Latin-1 file not read correctly'
        finally:
            shutil.rmtree(temp_dir)

    def test_encoding_detection(self):
        temp_dir = tempfile.mkdtemp()
        try:
            utf8_bom_file = os.path.join(temp_dir, 'utf8_bom_test.txt')
            with open(utf8_bom_file, 'wb') as f:
                f.write(b'\xef\xbb\xbf')
                f.write('这是带BOM的UTF-8文件'.encode('utf-8'))

            utf16_file = os.path.join(temp_dir, 'utf16_test.txt')
            with open(utf16_file, 'w', encoding='utf-16-le') as f:
                f.write('这是UTF-16 LE编码文件')

            reader = TxtReader(auto_detect_encoding=True)

            detected_encoding = reader.detect_encoding(utf8_bom_file)
            assert detected_encoding == 'utf-8-sig', f'Expected utf-8-sig, got {detected_encoding}'

            docs = reader._load_data(utf8_bom_file)
            assert len(docs) == 1
            assert 'UTF-8' in docs[0].text
        finally:
            shutil.rmtree(temp_dir)

    def test_encoding_cache(self):
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_dir, 'cache_test.txt')
            with open(test_file, 'w', encoding='gbk') as f:
                f.write('测试编码缓存功能')

            TxtReader.clear_encoding_cache()

            reader = TxtReader(auto_detect_encoding=True, use_encoding_cache=True)
            encoding1 = reader.detect_encoding(test_file, use_cache=True)

            cache_stats = TxtReader.get_encoding_cache_stats()
            assert cache_stats['cache_size'] > 0, 'Cache should contain entries'

            encoding2 = reader.detect_encoding(test_file, use_cache=True)
            assert encoding1 == encoding2, 'Cached encoding should match'

            encoding3 = reader.detect_encoding(test_file, use_cache=False)
            assert encoding3 == encoding1, 'Encoding should be consistent'

            TxtReader.clear_encoding_cache()
            cache_stats = TxtReader.get_encoding_cache_stats()
            assert cache_stats['cache_size'] == 0, 'Cache should be empty after clear'
        finally:
            shutil.rmtree(temp_dir)

    def test_reader_with_special_encodings(self):
        temp_dir = tempfile.mkdtemp()
        try:
            special_chars_file = os.path.join(temp_dir, 'special_chars.txt')
            with open(special_chars_file, 'w', encoding='utf-8') as f:
                f.write('特殊字符测试：©®™€£¥§¶†‡\n中文标点：，。！？；：""''《》【】\nEmoji: 😀🎉🔥💯')

            mixed_lang_file = os.path.join(temp_dir, 'mixed_lang.txt')
            with open(mixed_lang_file, 'w', encoding='utf-8') as f:
                f.write('English text\n中文文本\n日本語テキスト\n한국어 텍스트\nРусский текст')

            reader = SimpleDirectoryReader(input_dir=temp_dir)
            docs = []
            for doc in reader():
                docs.append(doc)

            assert len(docs) == 2, f'Expected 2 docs, got {len(docs)}'

            texts = [doc.text for doc in docs]
            assert any('©®™' in text for text in texts), 'Special characters not read correctly'
            assert any('日本語' in text for text in texts), 'Mixed language content not read correctly'
        finally:
            shutil.rmtree(temp_dir)
