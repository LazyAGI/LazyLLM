import lazyllm
from lazyllm.tools.rag.transform import (
    SentenceSplitter, CharacterSplitter, RecursiveSplitter,
    _TextSplitterBase, _Split, _TokenTextSplitter,
)
from lazyllm.tools.rag.doc_node import DocNode
import pytest
from unittest.mock import MagicMock
from lazyllm.tools.rag.document import Document
from lazyllm.tools.rag.retriever import Retriever

@pytest.fixture
def doc_node():
    node = MagicMock(spec=DocNode)
    node.get_text.return_value = '这是一个测试文本，用于验证分割器的逻辑是否正确。请认真检查！'
    node.get_metadata_str.return_value = ''
    return node


class TestSentenceSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the SentenceSplitter.'''
        self.splitter = SentenceSplitter(chunk_size=30, chunk_overlap=10)

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]

        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            "Before college the two main things I worked on, outside of school, were writing and programming.I didn't write essays.",  # noqa: E501
            "I didn't write essays.I wrote what beginning writers were supposed to write then, and probably still are: short stories.My stories were awful.",  # noqa: E501
            'My stories were awful.They had hardly any plot, just characters with strong feelings, which I imagined made them deep.',  # noqa: E501
        ]
        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_split(self):
        splitter = SentenceSplitter(chunk_size=10, chunk_overlap=0)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter._split(text, chunk_size=10)
        assert len(splits) == 2
        assert splits[0].text == 'This is a test sentence.'
        assert splits[1].text == 'It needs to be split into multiple chunks.'

    def test_merge(self):
        splitter = SentenceSplitter(chunk_size=15, chunk_overlap=10)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter._split(text, chunk_size=15)
        chunks = splitter._merge(splits, chunk_size=15)
        assert chunks == ['This is a test sentence. It needs to be split into multiple chunks.']

    def test_split_text(self):
        splitter = SentenceSplitter(chunk_size=10, chunk_overlap=0)
        text = 'This is a test sentence. It needs to be split into multiple chunks.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['This is a test sentence.', 'It needs to be split into multiple chunks.']


class TestCharacterSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the CharacterSplitter.'''
        self.splitter = CharacterSplitter(chunk_size=30, overlap=10, separator=',', keep_separator=True)

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]
        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            ' Before college the two main things I worked on, outside of school,',
            " outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then,",  # noqa: E501
            ' wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot,',  # noqa: E501
            ' stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'  # noqa: E501
        ]
        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_get_separator_pattern(self):
        splitter = CharacterSplitter(separator=' ')
        sep_pattern = splitter._get_separator_pattern(' ')
        assert sep_pattern == r'(?: )'
        splitter = CharacterSplitter(separator=' ', keep_separator=True)
        sep_pattern = splitter._get_separator_pattern(' ')
        assert sep_pattern == r'( )'

    def test_default_split(self):
        splitter = CharacterSplitter(separator=' ')
        text = 'Hello, world! This is a test.'
        splits = splitter.default_split(splitter._get_separator_pattern(' '), text)
        assert splits == ['Hello,', 'world!', 'This', 'is', 'a', 'test.']

        splitter = CharacterSplitter(separator=',', keep_separator=True)
        text = 'Hello, world! This is a test.'
        splits = splitter.default_split(splitter._get_separator_pattern(','), text)
        assert splits == ['Hello,', ' world! This is a test.']

        splitter = CharacterSplitter(separator=',', keep_separator=False)
        text = 'Hello, world! This is a test.'
        splits = splitter.default_split(splitter._get_separator_pattern(','), text)
        assert splits == ['Hello', ' world! This is a test.']

    def test_split_text(self):
        splitter = CharacterSplitter(separator=',', chunk_size=7, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']

    def test_split_text_with_split_fn(self):
        splitter = CharacterSplitter(separator=',', chunk_size=7, overlap=0)
        splitter.set_split_fns([lambda t: t.split(',')])
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']
        splitter.add_split_fn(lambda t: t.split(' '), index=0)
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello,', 'world!', 'This', 'is', 'a', 'test.']
        splitter.clear_split_fns()
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', ' world! This is a test.']

    def test_split_text_with_weak_split_fn(self):
        splitter = CharacterSplitter(separator=',', chunk_size=4, overlap=0)
        splitter.set_split_fns([lambda t: t.split('!')])
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        print(splits)
        assert splits == ['Hello, world', ' This is a test', '.']


class TestRecursiveSplitter:
    def setup_method(self):
        '''Setup for tests: initialize the RecursiveSplitter.'''
        self.splitter = RecursiveSplitter(chunk_size=30, overlap=10, separators=[',', '.', ' ', ''])

    def test_forward(self):
        text = ''' Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.'''  # noqa: E501
        docs = [DocNode(text=text)]
        result = self.splitter.batch_forward(docs, node_group='default')
        result_texts = [n.get_text() for n in result]
        expected_texts = [
            ' Before college the two main things I worked on outside of school',
            " outside of school were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then",  # noqa: E501
            ' I wrote what beginning writers were supposed to write then and probably still are: short stories. My stories were awful. They had hardly any plot',  # noqa: E501
            ' My stories were awful. They had hardly any plot just characters with strong feelings which I imagined made them deep.'  # noqa: E501
        ]

        assert result_texts == expected_texts

        trans = lazyllm.pipeline(lambda x: x, self.splitter)
        assert [n.get_text() for n in trans(docs[0])] == expected_texts

    def test_split_text(self):
        splitter = RecursiveSplitter(separators=['\n\n', '\n', '!', ' '], chunk_size=5, overlap=0)
        text = 'Hello\n\nworld! This\nis a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This', 'is a test.']

    def test_split_text_with_character_split_fn(self):
        splitter = RecursiveSplitter(separators=['\n\n', '\n', '!', ' '], chunk_size=9, overlap=0)
        splitter.set_split_fns([lambda t: t.split('\n\n')])
        text = 'Hello\n\nworld! This\nis a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']
        splitter.add_split_fn(lambda t: t.split('\n'))
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']
        splitter.clear_split_fns()
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello', 'world! This\nis a test.']


class TestTextSplitterBase:
    def test_token_size(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        token_size = splitter._token_size(text)
        assert token_size == 9

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError):
            _TextSplitterBase(chunk_size=2, overlap=10)

    def test_split(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter._split(text, chunk_size=5)
        assert splits == [
            _Split(text='Hello, world!', is_sentence=True, token_size=4),
            _Split(text='This is a test.', is_sentence=True, token_size=5),
        ]

    def test_merge(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        splits = [
            _Split(text='Hello, world!', is_sentence=True, token_size=4),
            _Split(text='This is a test.', is_sentence=True, token_size=5),
        ]
        merged = splitter._merge(splits, chunk_size=5)
        assert merged == ['Hello, world!', 'This is a test.']

    def test_split_text(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits = splitter.split_text(text, metadata_size=0)
        assert splits == ['Hello, world!', 'This is a test.']

    def test_empty_text(self):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        chunks = splitter.split_text('', metadata_size=0)
        assert chunks == ['']

    def test_overlap_behavior(self):
        splitter = _TextSplitterBase(chunk_size=10, overlap=2)
        text = 'abcdefghijabcdefghij'
        splits = [
            _Split(text[:10], is_sentence=True, token_size=len(splitter.token_encoder(text[:10]))),
            _Split(text[10:14], is_sentence=True, token_size=len(splitter.token_encoder(text[10:14]))),
            _Split(text[14:], is_sentence=True, token_size=len(splitter.token_encoder(text[14:])))
        ]
        chunks = splitter._merge(splits, chunk_size=10)
        assert len(chunks) >= 2
        assert splitter.token_encoder(chunks[0])[-2:] == splitter.token_encoder(chunks[1])[:2]

    def test_metadata_size_limit(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        doc_node.get_metadata_str.return_value = 'x' * 100
        with pytest.raises(ValueError):
            splitter.split_text('短文本', metadata_size=200)

    def test_get_splits_by_fns(self):
        splitter = _TextSplitterBase(chunk_size=5, overlap=0)
        text = 'Hello, world! This is a test.'
        splits, is_sentence = splitter._get_splits_by_fns(text)
        assert splits == ['Hello, world!', 'This is a test.']
        assert is_sentence is True

    def test_get_metadata_size(self):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        node = DocNode(text='Hello, world! This is a test.')
        metadata_size = splitter._get_metadata_size(node)
        assert metadata_size == 0

    def test_transform_returns_chunks(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        chunks = splitter.transform(doc_node)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_batch_forward_single(self, doc_node):
        splitter = _TextSplitterBase(chunk_size=20, overlap=10)
        doc_node.children = {}
        result = splitter.batch_forward(doc_node, node_group='test')
        assert isinstance(result, list)
        assert all(hasattr(c, 'text') for c in result)
        assert 'test' in doc_node.children

    def test_from_tiktoken_encoder_basic(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(encoding_name='gpt2')
        assert result is splitter
        assert splitter.token_encoder is not None
        assert splitter.token_decoder is not None

    def test_from_tiktoken_encoder_with_model_name(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(model_name='gpt-3.5-turbo')

        assert result is splitter
        assert splitter.token_encoder is not None
        assert splitter.token_decoder is not None

    def test_from_tiktoken_encoder_encoding_decoding(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')
        text = 'Hello, world!'
        encoded = splitter.token_encoder(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0

    def test_from_tiktoken_encoder_token_size(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'This is a test sentence.'
        token_size = splitter._token_size(text)

        assert isinstance(token_size, int)
        assert token_size > 0

    def test_from_tiktoken_encoder_chaining(self):
        splitter = _TextSplitterBase(chunk_size=100, overlap=20)

        result = splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'Test text'
        chunks = result.split_text(text, metadata_size=0)
        assert chunks == ['Test text']


class TestTokenTextSplitter:
    def test_token_splitter_basic(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'hello world'
        splits = token_splitter._split(text, chunk_size=10)
        assert isinstance(splits, list)
        assert all(isinstance(s, _Split) for s in splits)

    def test_token_splitter_overlap_behavior(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijabcdefghij'
        splits = token_splitter._split(text, chunk_size=10)
        chunks = token_splitter._merge(splits, chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == 'abcdefghijabcdefghij'

    def test_token_splitter_exact_overlap(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijabcdefghij'
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) >= 1
        assert chunks[0] == 'abcdefghijabcdefghij'

    def test_token_splitter_short_text(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'short'
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) == 1
        assert chunks[0] == 'short'

    def test_token_splitter_exact_chunk_size(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'a' * 10
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_token_splitter_large_text(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'a' * 50
        chunks = token_splitter.split_text(text, metadata_size=0)
        assert len(chunks) > 1

        for i in range(len(chunks) - 1):
            assert chunks[i][-3:] == chunks[i + 1][:3]

    def test_token_splitter_merge_returns_text_only(self):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        text = 'abcdefghijklmnopqrst'
        splits = token_splitter._split(text, chunk_size=10)
        merged = token_splitter._merge(splits, chunk_size=10)

        assert isinstance(merged, list)
        assert all(isinstance(m, str) for m in merged)

    def test_token_splitter_transform_with_docnode(self, doc_node):
        token_splitter = _TokenTextSplitter(chunk_size=10, overlap=3)
        chunks = token_splitter.transform(doc_node)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_token_text_splitter_with_tiktoken(self):
        splitter = _TokenTextSplitter(chunk_size=10, overlap=5)
        splitter.from_tiktoken_encoder(encoding_name='gpt2')

        text = 'This is a test sentence that needs to be split into multiple chunks.'
        chunks = splitter.split_text(text, metadata_size=0)

        assert isinstance(chunks, list)
        assert len(chunks) > 1

        for i in range(len(chunks) - 1):
            tokens1 = splitter.token_encoder(chunks[i])
            tokens2 = splitter.token_encoder(chunks[i + 1])

            overlap_size = min(5, len(tokens1), len(tokens2))
            if overlap_size > 0:
                assert tokens1[-overlap_size:] == tokens2[:overlap_size]


class TestDocumentSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        document.create_node_group(
            name='character_test',
            transform=CharacterSplitter,
            chunk_size=128,
            overlap=0,
            separator=',',
            keep_separator=True
        )
        document.create_node_group(
            name='recursive_test',
            transform=RecursiveSplitter,
            chunk_size=128,
            overlap=0,
            separators=['\n\n', '\n', '.', ' ']
        )
        llm = lazyllm.OnlineChatModule(source='qwen')

        prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
        llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
        query = '何为天道？'

        self.document = document
        self.llm = llm
        self.query = query

    def test_sentence_split(self):
        document = self.document
        document.activate_groups('sentence_test')
        document.start()
        retriever = Retriever(document, group_name='sentence_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None

    def test_recursive_split(self):
        document = self.document
        document.activate_groups('recursive_test')
        document.start()
        retriever = Retriever(document, group_name='recursive_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None


class TestDocumentChainSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        document.create_node_group(
            name='recursive_test',
            transform=RecursiveSplitter,
            chunk_size=128,
            overlap=0,
            parent='sentence_test'
        )
        document.create_node_group(
            name='character_test',
            transform=CharacterSplitter,
            chunk_size=128,
            overlap=0,
            separator=' ',
            parent='recursive_test'
        )
        llm = lazyllm.OnlineChatModule(source='qwen')
        prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
        llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
        query = '何为天道？'

        self.document = document
        self.llm = llm
        self.query = query

    def test_sentence_split(self):
        document = self.document
        document.activate_groups('sentence_test')
        document.start()
        retriever = Retriever(document, group_name='sentence_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None

    def test_recursive_split(self):
        document = self.document
        document.activate_groups('recursive_test')
        document.start()
        retriever = Retriever(document, group_name='recursive_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None


class TestDIYDocumentSplit:
    def setup_method(self):
        document = Document(
            dataset_path='rag_master',
            manager=False
        )
        document.create_node_group(
            name='sentence_test',
            transform=SentenceSplitter,
            chunk_size=128,
            chunk_overlap=10
        )
        splitter = CharacterSplitter(chunk_size=128, overlap=10, separator=' ')
        splitter.set_split_fns([lambda x: x.split(' ')])
        document.create_node_group(
            name='character_test',
            transform=splitter,
            parent='sentence_test')

        llm = lazyllm.OnlineChatModule(source='qwen')
        prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
        llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
        query = '何为天道？'

        self.document = document
        self.llm = llm
        self.query = query

    def test_character_split(self):
        document = self.document
        document.activate_groups('character_test')
        document.start()
        retriever = Retriever(document, group_name='character_test', similarity='bm25', topk=3)
        doc_node_list = retriever(query=self.query)
        assert len(doc_node_list) == 3
        res = self.llm({
            'query': self.query,
            'context_str': ''.join([node.get_content() for node in doc_node_list]),
        })
        assert res is not None
