import unittest
from unittest.mock import MagicMock
from lazyllm import LLMParser, TrainableModule


class TestLLMParser(unittest.TestCase):
    def setUp(self):
        self.llm = TrainableModule("internlm2-chat-7b")
        self.llm.start()

        self.mock_node = MagicMock()
        self.mock_node.get_text.return_value = (
            "Hello, I am an AI robot developed by SenseTime, named LazyLLM. "
            "My mission is to assist you in building the most powerful large-scale model applications with minimal cost."
        )

        self.summary_parser = LLMParser(self.llm, language="en", task_type="summary")
        self.keywords_parser = LLMParser(self.llm, language="en", task_type="keywords")

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
