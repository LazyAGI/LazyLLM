import unittest
from unittest.mock import MagicMock
from lazyllm import LLMParser, TrainableModule
from lazyllm.launcher import cleanup
from lazyllm.tools.rag import DocNode
from lazyllm.engine import LightEngine

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
        cls.qa_parser = LLMParser(cls.llm, language="en", task_type="qa")

    @classmethod
    def teardown_class(cls):
        cleanup()

    def test_summary_transform(self):
        result = self.summary_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert len(result[0]) < 150
        assert "LazyLLM" in result[0]

    def test_keywords_transform(self):
        result = self.keywords_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert 1 < len(result) < 10
        assert isinstance(result[0], str)
        assert len(result[0]) < 20
        assert "LazyLLM" in result

    def test_qa_transform(self):
        result = self.qa_parser.transform(self.mock_node)
        assert isinstance(result, list)
        assert isinstance(result[0], DocNode)
        text, content = result[0].text, result[0].get_content()
        assert len(text) < len(content) and text in content
        assert 'query:' in content and 'answer' in content

    def test_QustionRewrite_and_ParameterExtractor(self):
        nodes = [
            dict(
                id="1",
                kind="QustionRewrite",
                name="m1",
                args=dict(
                    base_model=self.llm,
                    formatter="str",
                ),
            )
        ]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        res = engine.run(gid, "Model Context Protocol是啥")
        assert isinstance(res, str)
        engine.reset()

        nodes = [
            dict(
                id="1",
                kind="QustionRewrite",
                name="m2",
                args=dict(
                    base_model=self.llm,
                    formatter="list",
                ),
            )
        ]
        gid = engine.start(nodes, edges)
        res = engine.run(gid, "RAG是什么？")
        assert isinstance(res, list) and len(res) > 0
        engine.reset()

        nodes = [
            dict(
                id="1",
                kind="ParameterExtractor",
                name="m1",
                args=dict(
                    base_model=self.llm,
                    param=["year"],
                    types=["int"],
                    description=["年份"],
                    require=[True],
                ),
            )
        ]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]
        gid = engine.start(nodes, edges)
        input = "This year is 2023"
        res = engine.run(gid, input)
        assert isinstance(res, dict)
        assert "year" in res
        assert isinstance(res["year"], int)
        assert res["year"] == 2023

if __name__ == "__main__":
    unittest.main()
