from lazyllm.tools import HttpTool

class TestHttpTool(object):
    def test_forward(self):
        code_str = "def identity(content): return content"
        tool = HttpTool(method='GET', url='http://www.baidu.com/', code_str=code_str)
        ret = tool()
        assert '百度' in ret['content']

    def test_without_args(self):
        tool = HttpTool()
        assert tool() is None

    def test_no_url(self):
        code_str = "def echo(s): return s"
        tool = HttpTool(code_str=code_str)
        content = "hello, world!"
        assert tool(content) == content

    def test_math(self):
        code_str = "def exp(v, n): return v ** n"
        tool = HttpTool(code_str=code_str)
        assert tool(v=10, n=2) == 100
