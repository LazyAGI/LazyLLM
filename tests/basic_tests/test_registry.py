from lazyllm.common import Register
from lazyllm.tools.agent.toolsManager import ModuleTool

def orig_func(self):
    pass


class TestRegistry:
    def setup_method(self):
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        reg = Register(ModuleTool, ["apply"])
        if reg:
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        self._reg = reg.new_group('group_for_test')
        if self._reg:

    def test_register(self):
        registered_func = self._reg(orig_func)
        assert registered_func == orig_func


    def test_register_with_new_func_name(self):
        new_func_name = 'another_func_name'
        registered_func = self._reg(orig_func, new_func_name)
        assert registered_func != orig_func
        assert registered_func.__name__ == new_func_name
