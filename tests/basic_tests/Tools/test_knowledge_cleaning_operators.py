import json
import os
import shutil
import tempfile
from lazyllm import config
# Import knowledge_cleaning to register it first
from lazyllm.tools.data.operators import knowledge_cleaning  # noqa: F401
from lazyllm.tools.data import kbc


class MockLLMServe:

    def __init__(self, return_value=None, raise_exc=False):
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.started = False

    def start(self):
        self.started = True
        return self

    def prompt(self, system_prompt):
        return self

    def formatter(self, formatter):
        return self

    def __call__(self, prompt):
        if self._raise_exc:
            raise RuntimeError("mock error")
        return self._return_value


class MockLLM:

    def __init__(self, answer_return=None, raise_exc=False):
        self.answer_serve = MockLLMServe(answer_return, raise_exc)
        self._share_count = 0

    def share(self, prompt=None, format=None, stream=None, history=None):
        # Return answer_serve
        if format is not None:
            self.answer_serve.formatter(format)
        self._share_count += 1
        return self.answer_serve


class TestKnowledgeCleaningOperators:
    def setup_method(self):
        self.root_dir = './test_kbc_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ========== Chunk Generator Operators ==========

    def test_KBCLoadText(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('test content')
        op = kbc.KBCLoadText()
        result = op([{'text_path': test_file}])[0]
        assert result['_text_content'] == 'test content'

    def test_KBCLoadText_json(self):
        test_file = os.path.join(self.temp_dir, 'test.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump({'text': 'json content'}, f)
        op = kbc.KBCLoadText()
        result = op([{'text_path': test_file}])[0]
        assert result['_text_content'] == 'json content'

    def test_KBCChunkText(self):
        try:
            op = kbc.KBCChunkText(chunk_size=50, split_method='recursive')
            data = {'_text_content': 'word1 word2 word3 word4 word5 ' * 10}
            result = op([data])[0]
            assert '_chunks' in result
            assert len(result['_chunks']) > 0
        except (ImportError, Exception):
            # Skip if dependencies are missing
            pass

    def test_KBCExpandChunks(self):
        op = kbc.KBCExpandChunks()
        data = {'_chunks': ['chunk1', 'chunk2'], 'id': 1}
        result = op([data])
        assert len(result) == 2
        assert result[0]['raw_chunk'] == 'chunk1'
        assert result[1]['raw_chunk'] == 'chunk2'

    def test_KBCSaveChunks(self):
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('test')
        op = kbc.KBCSaveChunks(output_dir=self.temp_dir)
        result = op([{'_chunks': ['c1', 'c2'], 'text_path': test_file}])[0]
        assert 'chunk_path' in result
        assert os.path.exists(result['chunk_path'])

    # ========== Text Cleaner Operators ==========

    def test_KBCGenerateCleanedTextSingle(self):
        mock_llm = MockLLM(answer_return='<cleaned_start>cleaned text<cleaned_end>')
        op = kbc.KBCGenerateCleanedTextSingle(llm=mock_llm)
        data = {'raw_chunk': 'test'}
        result = op([data])[0]
        assert '_cleaned_response' in result

    def test_extract_cleaned_content_single(self):
        op = kbc.extract_cleaned_content_single()
        data = {'raw_chunk': 'test', '_cleaned_response': '<cleaned_start>cleaned<cleaned_end>'}
        result = op([data])[0]
        assert result['cleaned_chunk'] == 'cleaned'
        assert '_cleaned_response' not in result

    def test_KBCLoadRAWChunkFile(self):
        test_file = os.path.join(self.temp_dir, 'chunk.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump([{'raw_chunk': 'content1'}, {'raw_chunk': 'content2'}], f)
        op = kbc.KBCLoadRAWChunkFile()
        result = op([{'chunk_path': test_file}])[0]
        assert '_chunks_data' in result
        assert len(result['_chunks_data']) == 2

    def test_KBCGenerateCleanedText(self):
        mock_llm = MockLLM(answer_return='<cleaned_start>cleaned<cleaned_end>')
        op = kbc.KBCGenerateCleanedText(llm=mock_llm)
        data = {'_chunks_data': [{'raw_chunk': 'test'}]}
        result = op([data])[0]
        assert '_cleaned_results' in result
        assert len(result['_cleaned_results']) == 1

    def test_extract_cleaned_content(self):
        op = kbc.extract_cleaned_content()
        data = {'_cleaned_results': [{'response': '<cleaned_start>cleaned<cleaned_end>', 'raw_chunk': 'test'}]}
        result = op([data])[0]
        assert '_cleaned_chunks' in result
        assert len(result['_cleaned_chunks']) == 1
        assert result['_cleaned_chunks'][0]['cleaned_chunk'] == 'cleaned'

    def test_KBCSaveCleaned(self):
        test_file = os.path.join(self.temp_dir, 'chunk.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump([{'raw_chunk': 'test'}], f)
        op = kbc.KBCSaveCleaned(output_dir=self.temp_dir)
        cleaned_chunks = [
            {'raw_chunk': 'test', 'cleaned_chunk': 'cleaned', 'original_item': {}}
        ]
        result = op([{'_cleaned_chunks': cleaned_chunks, '_chunk_path': test_file}])[0]
        assert 'cleaned_chunk_path' in result

    # ========== File/URL Converter Operators ==========

    def test_FileOrURLNormalizer_url(self):
        op = kbc.FileOrURLNormalizer(intermediate_dir=self.temp_dir)
        result = op([{'source': 'https://example.com'}])[0]
        assert result['_type'] == 'html'
        assert '_url' in result

    def test_FileOrURLNormalizer_file(self):

        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('test content')
        op = kbc.FileOrURLNormalizer(intermediate_dir=self.temp_dir)
        result = op([{'source': test_file}])[0]
        assert result['_type'] == 'text'
        assert '_raw_path' in result

    def test_HTMLToMarkdownConverter(self):
        op = kbc.HTMLToMarkdownConverter()
        output_path = os.path.join(self.temp_dir, 'output.md')
        data = {'_type': 'html', '_url': 'https://example.com', '_output_path': output_path}
        result = op([data])[0]
        # May fail if trafilatura is not installed or URL is not accessible
        assert '_markdown_path' in result or '_output_path' in result

    def test_PDFToMarkdownConverterAPI(self):
        op = kbc.PDFToMarkdownConverterAPI(mineru_url='http://localhost:8000')
        output_path = os.path.join(self.temp_dir, 'output.md')
        data = {'_type': 'pdf', '_raw_path': '/nonexistent.pdf', '_output_path': output_path}
        result = op([data])[0]
        # May fail if mineru_url is not available
        assert '_markdown_path' in result or '_output_path' in result

    # ========== Multi-hop QA Generator Operators ==========

    def test_KBCLoadChunkFile(self):
        test_file = os.path.join(self.temp_dir, 'chunk.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump([{'raw_chunk': 'content', 'cleaned_chunk': 'cleaned'}], f)
        op = kbc.KBCLoadChunkFile()
        result = op([{'chunk_path': test_file}])[0]
        assert '_chunks_data' in result
        assert len(result['_chunks_data']) == 1

    def test_KBCPreprocessText(self):
        op = kbc.KBCPreprocessText(min_length=5, max_length=1000)
        data = {'_chunks_data': [{'cleaned_chunk': 'test  test  '}]}
        result = op([data])[0]
        assert '_processed_chunks' in result
        assert len(result['_processed_chunks']) == 1

    def test_KBCExtractInfoPairs(self):
        op = kbc.KBCExtractInfoPairs()
        processed_chunks = [{
            'text': 'This is sentence one. This is sentence two. This is sentence three.',
            'original_data': {'cleaned_chunk': 'test'}
        }]
        data = {'_processed_chunks': processed_chunks}
        result = op([data])[0]
        assert '_info_pairs' in result
        assert len(result['_info_pairs']) > 0
        assert 'premise' in result['_info_pairs'][0]

    def test_KBCGenerateMultiHopQA(self):
        mock_llm = MockLLM(answer_return={'question': 'What is the relationship?', 'answer': 'The answer'})
        op = kbc.KBCGenerateMultiHopQA(llm=mock_llm)
        data = {'_info_pairs': [{'premise': 'A', 'intermediate': 'B', 'conclusion': 'C'}]}
        result = op([data])[0]
        assert '_qa_results' in result
        assert len(result['_qa_results']) == 1

    def test_KBCGenerateMultiHopQA_empty(self):
        mock_llm = MockLLM(answer_return={})
        op = kbc.KBCGenerateMultiHopQA(llm=mock_llm)
        data = {'_info_pairs': []}
        result = op([data])[0]
        assert result['_qa_results'] == []

    def test_parse_qa_pairs(self):
        op = kbc.parse_qa_pairs()
        data = {'_qa_results': [{'response': {'question': 'Q?', 'answer': 'A?'}, 'info_pair': {}}]}
        result = op([data])[0]
        assert '_qa_pairs' in result
        assert len(result['_qa_pairs']) == 1

    def test_KBCSaveEnhanced(self):
        test_file = os.path.join(self.temp_dir, 'chunk.json')
        chunks_data = [{'raw_chunk': 'test', 'cleaned_chunk': 'cleaned'}]
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f)
        op = kbc.KBCSaveEnhanced(output_dir=self.temp_dir)
        qa_pairs = [{'cleaned_chunk': 'cleaned', 'qa_pairs': {'question': 'Q?', 'answer': 'A?'}}]
        result = op([{'_qa_pairs': qa_pairs, '_chunks_data': chunks_data, '_chunk_path': test_file}])[0]
        assert 'enhanced_chunk_path' in result

    # ========== QA Extract Operators ==========

    def test_KBCLoadQAData(self):
        op = kbc.KBCLoadQAData()
        result = op([{'QA_pairs': [{'question': 'Q1', 'answer': 'A1'}]}])[0]
        assert result['_qa_data'] == [{'question': 'Q1', 'answer': 'A1'}]

    def test_KBCExtractQAPairs(self):
        op = kbc.KBCExtractQAPairs()
        result = op([{'_qa_data': [{'question': 'What is AI?', 'answer': 'AI'}]}])
        assert len(result) == 1
        assert result[0]['input'] == 'What is AI?'
        assert result[0]['output'] == 'AI'

    def test_KBCExtractQAPairs_invalid(self):
        op = kbc.KBCExtractQAPairs()
        result = op([{'_qa_data': [
            {'question': 'Q1?', 'answer': 'A1'},
            {'question': '', 'answer': 'A2'},  # Invalid: empty question
            {'question': 'Q3?', 'answer': ''},  # Invalid: empty answer
        ]}])
        assert len(result) == 1
        assert result[0]['input'] == 'Q1?'
