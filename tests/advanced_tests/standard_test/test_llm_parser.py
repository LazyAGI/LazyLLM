import unittest
from unittest.mock import MagicMock
from lazyllm import LLMParser, TrainableModule, deploy
from lazyllm.launcher import cleanup
from lazyllm.tools.rag import DocNode


class TestLLMParser(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.llm = TrainableModule('internlm2-chat-7b').deploy_method(deploy.vllm).start()
        cls.mock_node = MagicMock()
        cls.mock_node.get_text.return_value = (
            'Hello, I am an AI robot developed by SenseTime, named LazyLLM. '
            'My mission is to assist you in building the most powerful large-scale model applications with minimal cost.'
        )

        cls.summary_parser = LLMParser(cls.llm, language='en', task_type='summary')
        cls.keywords_parser = LLMParser(cls.llm, language='en', task_type='keywords')
        cls.qa_parser = LLMParser(cls.llm, language='en', task_type='qa')

    @classmethod
    def teardown_class(cls):
        cleanup()

    def test_summary_transform(self):
        result = self.summary_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert len(result[0]) < 150
        assert 'LazyLLM' in result[0]

    def test_keywords_transform(self):
        result = self.keywords_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert 1 < len(result) < 10
        assert isinstance(result[0], str)
        assert len(result[0]) < 20
        assert 'LazyLLM' in result

    def test_qa_transform(self):
        result = self.qa_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], DocNode)
        text, content = result[0].text, result[0].get_content()
        assert len(text) < len(content) and text in content
        assert 'query:' in content and 'answer' in content
