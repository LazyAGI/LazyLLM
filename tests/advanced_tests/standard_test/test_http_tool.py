from lazyllm.tools import HttpTool

class TestHttpTool(object):
    def setup_method(self):
        code_str = "def identity(content): return content"
        self._tool = HttpTool(method='GET', url='http://www.baidu.com/', post_process_code=code_str)

    def test_forward(self):
        ret = self._tool()
        print(ret['content'])
        assert '百度' in ret['content']
