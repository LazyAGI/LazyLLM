"""Test cases for AgenticRAG operators"""
import pytest
from lazyllm.tools.data import (
    AgenticRAGQAF1SampleEvaluator,
    AgenticRAGAtomicTaskGenerator,
    AgenticRAGDepthQAGenerator,
    AgenticRAGWidthQAGenerator,
)


class TestAgenticRAGQAF1SampleEvaluator:
    """Test AgenticRAGQAF1SampleEvaluator"""

    def setup_method(self):
        self.evaluator = AgenticRAGQAF1SampleEvaluator(_save_data=False)

    def test_init(self):
        """Test initialization"""
        assert self.evaluator is not None

    def test_call_with_list_perfect_match(self):
        """Test __call__ with list input - perfect match"""
        data = [
            {"refined_answer": "hello world", "golden_doc_answer": "hello world"},
        ]
        results = self.evaluator(data)
        assert len(results) == 1
        assert "F1Score" in results[0]
        assert results[0]["F1Score"] == 1.0

    def test_call_with_list_partial_match(self):
        """Test __call__ with list input - partial match"""
        data = [
            {"refined_answer": "quick brown fox", "golden_doc_answer": "quick brown dog"},
        ]
        results = self.evaluator(data)
        assert len(results) == 1
        assert "F1Score" in results[0]
        assert 0 < results[0]["F1Score"] < 1.0

    def test_call_with_list_no_match(self):
        """Test __call__ with list input - no match"""
        data = [
            {"refined_answer": "apple orange", "golden_doc_answer": "cat dog"},
        ]
        results = self.evaluator(data)
        assert len(results) == 1
        assert "F1Score" in results[0]
        assert results[0]["F1Score"] == 0.0

    def test_call_with_multiple_items(self):
        """Test __call__ with multiple items"""
        data = [
            {"refined_answer": "hello world", "golden_doc_answer": "hello world"},
            {"refined_answer": "foo bar", "golden_doc_answer": "foo baz"},
            {"refined_answer": "apple", "golden_doc_answer": "orange"},
        ]
        results = self.evaluator(data)
        assert len(results) == 3
        assert all("F1Score" in r for r in results)
        # First should be perfect match
        assert results[0]["F1Score"] == 1.0
        # Last should be no match
        assert results[2]["F1Score"] == 0.0

    def test_call_with_none_values(self):
        """Test __call__ with None values"""
        data = [
            {"refined_answer": None, "golden_doc_answer": "test"},
            {"refined_answer": "test", "golden_doc_answer": None},
        ]
        results = self.evaluator(data)
        assert len(results) == 2
        assert results[0]["F1Score"] == 0.0
        assert results[1]["F1Score"] == 0.0

    def test_call_with_multiple_ground_truths(self):
        """Test __call__ with multiple ground truths (list)"""
        data = [
            {
                "refined_answer": "the answer is yes",
                "golden_doc_answer": ["wrong answer", "the answer is yes", "another wrong"]
            },
        ]
        results = self.evaluator(data)
        assert len(results) == 1
        assert results[0]["F1Score"] == 1.0


class TestAgenticRAGAtomicTaskGenerator:
    """Test AgenticRAGAtomicTaskGenerator"""

    def test_init(self):
        """Test initialization without LLM serving"""
        generator = AgenticRAGAtomicTaskGenerator(
            llm_serving=None,
            data_num=50,
            max_per_task=5,
            max_question=5,
            _save_data=False
        )
        assert generator is not None


class TestAgenticRAGDepthQAGenerator:
    """Test AgenticRAGDepthQAGenerator"""

    def test_init(self):
        """Test initialization"""
        generator = AgenticRAGDepthQAGenerator(
            llm_serving=None,
            n_rounds=3,
            _save_data=False
        )
        assert generator is not None


class TestAgenticRAGWidthQAGenerator:
    """Test AgenticRAGWidthQAGenerator"""

    def test_init(self):
        """Test initialization"""
        generator = AgenticRAGWidthQAGenerator(
            llm_serving=None,
            _save_data=False
        )
        assert generator is not None
