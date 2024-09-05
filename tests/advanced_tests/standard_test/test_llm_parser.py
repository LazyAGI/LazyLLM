import unittest
from unittest.mock import MagicMock
from lazyllm import LLMParser, TrainableModule
from lazyllm.launcher import cleanup


class TestLLMParser(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.llm = TrainableModule("internlm2-chat-7b").start()
        cls.mock_node = MagicMock()
        cls.mock_node.get_text.return_value = (
            "Hello, I am an AI robot developed by SenseTime, named LazyLLM. "
            "My mission is to assist you in building the most powerful large-scale model applications with minimal cost."
        )

        cls.summary_parser = LLMParser(cls.llm, language="en", task_type="summary")
        cls.keywords_parser = LLMParser(cls.llm, language="en", task_type="keywords")

    @classmethod
    def teardown_class(cls):
        cleanup()

    def test_summary_transform(self):
        result = self.summary_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert len(result[0]) < 100
        assert "LazyLLM" in result[0]

    def test_keywords_transform(self):
        result = self.keywords_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert len(result[0]) < 100
        assert "LazyLLM," in result[0]


if __name__ == "__main__":
    unittest.main()
