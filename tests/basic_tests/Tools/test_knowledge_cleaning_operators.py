"""Test cases for Knowledge Cleaning operators"""
import pytest
import json
import tempfile
import os
from lazyllm.tools.data import (
    KBCChunkGenerator,
    KBCChunkGeneratorBatch,
    KBCTextCleaner,
    KBCTextCleanerBatch,
    FileOrURLToMarkdownConverterBatch,
    FileOrURLToMarkdownConverterAPI,
    KBCMultiHopQAGeneratorBatch,
    QAExtractor,
)


class TestKBCChunkGenerator:
    """Test KBCChunkGenerator"""

    def test_init_token_method(self):
        """Test initialization with token method"""
        generator = KBCChunkGenerator(
            chunk_size=256,
            chunk_overlap=25,
            split_method="token",
            tokenizer_name="bert-base-uncased",
            _save_data=False
        )
        assert generator is not None

    def test_init_sentence_method(self):
        """Test initialization with sentence method"""
        generator = KBCChunkGenerator(
            split_method="sentence",
            chunk_size=512,
            _save_data=False
        )
        assert generator is not None

    def test_init_semantic_method(self):
        """Test initialization with semantic method"""
        generator = KBCChunkGenerator(
            split_method="semantic",
            chunk_size=512,
            _save_data=False
        )
        assert generator is not None

    def test_init_recursive_method(self):
        """Test initialization with recursive method"""
        generator = KBCChunkGenerator(
            split_method="recursive",
            chunk_size=512,
            _save_data=False
        )
        assert generator is not None


class TestKBCChunkGeneratorBatch:
    """Test KBCChunkGeneratorBatch"""

    def test_init(self):
        """Test initialization"""
        generator = KBCChunkGeneratorBatch(
            chunk_size=256,
            split_method="token",
            _save_data=False
        )
        assert generator is not None


class TestKBCTextCleaner:
    """Test KBCTextCleaner"""

    def test_init(self):
        """Test initialization"""
        cleaner = KBCTextCleaner(
            llm_serving=None,
            lang="zh",
            _save_data=False
        )
        assert cleaner is not None

    def test_init_english(self):
        """Test initialization with English"""
        cleaner = KBCTextCleaner(
            llm_serving=None,
            lang="en",
            _save_data=False
        )
        assert cleaner is not None


class TestKBCTextCleanerBatch:
    """Test KBCTextCleanerBatch"""

    def test_init(self):
        """Test initialization"""
        cleaner = KBCTextCleanerBatch(
            llm_serving=None,
            lang="zh",
            _save_data=False
        )
        assert cleaner is not None


class TestFileOrURLToMarkdownConverterBatch:
    """Test FileOrURLToMarkdownConverterBatch"""

    def test_init(self):
        """Test initialization"""
        converter = FileOrURLToMarkdownConverterBatch(
            intermediate_dir="./test_cache",
            _save_data=False
        )
        assert converter is not None


class TestFileOrURLToMarkdownConverterAPI:
    """Test FileOrURLToMarkdownConverterAPI"""

    def test_init(self):
        """Test initialization"""
        converter = FileOrURLToMarkdownConverterAPI(
            intermediate_dir="./test_api_cache",
            mineru_backend="vlm",
            _save_data=False
        )
        assert converter is not None


class TestKBCMultiHopQAGeneratorBatch:
    """Test KBCMultiHopQAGeneratorBatch"""

    def test_init(self):
        """Test initialization"""
        generator = KBCMultiHopQAGeneratorBatch(
            llm_serving=None,
            lang="zh",
            _save_data=False
        )
        assert generator is not None


class TestQAExtractor:
    """Test QAExtractor"""

    def test_init(self):
        """Test initialization"""
        extractor = QAExtractor(
            input_qa_key="QA_pairs",
            input_instruction="Answer the question.",
            _save_data=False
        )
        assert extractor is not None

    def test_call_with_qa_pairs(self):
        """Test __call__ with QA_pairs data"""
        extractor = QAExtractor(_save_data=False)
        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "What is Python?", "answer": "A programming language."},
                        {"question": "What is LazyLLM?", "answer": "A framework for LLM."},
                    ]
                }
            },
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "What is AI?", "answer": "Artificial Intelligence."},
                    ]
                }
            }
        ]
        results = extractor(data)
        assert len(results) == 3
        assert all("instruction" in r for r in results)
        assert all("input" in r for r in results)
        assert all("output" in r for r in results)

    def test_call_empty_qa_pairs(self):
        """Test __call__ with empty QA_pairs"""
        extractor = QAExtractor(_save_data=False)
        data = [
            {"QA_pairs": {"qa_pairs": []}},
        ]
        results = extractor(data)
        assert len(results) == 0

    def test_call_invalid_qa_format(self):
        """Test __call__ with invalid QA format (missing question/answer)"""
        extractor = QAExtractor(_save_data=False)
        data = [
            {
                "QA_pairs": {
                    "qa_pairs": [
                        {"question": "", "answer": "Test A."},  # Empty question
                        {"question": "Test Q?", "answer": ""},  # Empty answer
                        {"question": "Valid Q?", "answer": "Valid A."},  # Valid
                    ]
                }
            }
        ]
        results = extractor(data)
        # Only valid QA pairs should be extracted
        assert len(results) == 1
        assert results[0]["input"] == "Valid Q?"

    def test_call_with_output_file(self):
        """Test __call__ with output JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.json")
            extractor = QAExtractor(output_json_file=output_file, _save_data=False)

            data = [
                {
                    "QA_pairs": {
                        "qa_pairs": [
                            {"question": "Q1?", "answer": "A1."},
                        ]
                    }
                }
            ]
            results = extractor(data)

            # Check file was created
            assert os.path.exists(output_file)

            # Check file contents
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert len(saved_data) == 1
            assert saved_data[0]["input"] == "Q1?"
