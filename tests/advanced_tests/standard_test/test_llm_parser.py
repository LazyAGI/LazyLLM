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
        rewrite_prompt = "你是一个查询重写助手，将用户查询分解为多个角度的具体问题。\
                注意，你不需要对问题进行回答，只需要根据问题的字面意思进行子问题拆分，输出不要超过 3 条.\
                下面是一个简单的例子：\
                输入：RAG是什么？\
                输出：RAG的定义是什么？\
                    RAG是什么领域内的名词？\
                    RAG有什么特点？\
                    \
                用户输入为："
        nodes = [
            dict(
                id="1",
                kind="QustionRewrite",
                name="m1",
                args=dict(
                    base_model=self.llm,
                    rewrite_prompt=rewrite_prompt,
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
                    rewrite_prompt=rewrite_prompt,
                    formatter="list",
                ),
            )
        ]
        gid = engine.start(nodes, edges)
        res = engine.run(gid, "RAG是什么？")
        assert isinstance(res, list) and len(res) > 0
        engine.reset()
        prompt = """
        你是一个智能助手，你的任务是从用户的输入中提取参数，并将其转换为json格式。
        ## 你需要根据提供的如下参数list，从文本提取相应的内容到json文件中。其中name是参数的名称，type是参数的类型，description是参数的说明。提取的参数名字、类型、说明如下：
        '''[{{"name": "year", "type": "int", "description": "年份","require"：true}}]'''
        ## 用户的指令如下：
        ''' '''

        ## 提取要求如下
        1. 你需要从用户的输入中提取出这些参数，并将其转换为json格式。请注意，json格式的键是参数的名称，值是提取到的值。请确保json格式正确，并且所有参数都被提取出来。
        2. 如果用户的输入中没有提到某个参数，请将其值设置为null；如果在提取要求中设置了"require"：true，却没有提取到该字段，请在返回的json中设置__is_success为0，__reason设置失败原因。
        __is_success字段解释：表示是否成功，成功时值为 1，失败时值为 0。
        3. 若提供的参数说明为：[{{"name": "year", "type": "int", "description": "年份","require"：true}}]，你可以使用以下的json格式来返回结果：
        {{"year": 2023,"__is_success"：1,"__reason":"提取成功"}}
        4. 仅输出json字符串，不要输出其他任何内容；输出的json格式需要使用双引号，冒号使用英文冒号
        5. 结合用户的输入和参数说明，尽量提取出更多的参数。
        """
        nodes = [
            dict(
                id="1",
                kind="ParameterExtractor",
                name="m1",
                args=dict(
                    base_model=self.llm,
                    param=["year"],
                    types=["int"],
                    prompt=prompt,
                ),
            )
        ]
        edges = [dict(iid="__start__", oid="1"), dict(iid="1", oid="__end__")]

        engine = LightEngine()
        gid = engine.start(nodes, edges)
        input = "This year is 2023"
        res = engine.run(gid, input)
        assert isinstance(res, dict)
        assert "year" in res
        assert isinstance(res["year"], int)
        assert res["year"] == 2023

if __name__ == "__main__":
    unittest.main()
