from lazyllm.module import ModuleBase
from lazyllm.tools import Tool

class Echo(ModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0]


class TestTool(object):
    def setup_method(self):
        code_str = "def add_one(v): return v + 1"
        self.tool = Tool(Echo(), code_str)

    def test_forward(self):
        assert self.tool(1) == 2
        assert self.tool(100) == 101
