from lazyllm.tools import fc_register

def orig_func(self):
    pass


class TestRegistry:
    def test_register(self):
        registered_func = fc_register('tool')(orig_func)
        assert registered_func == orig_func

    def test_register_with_new_func_name(self):
        new_func_name = 'another_func_name'
        registered_func = fc_register('tool')(orig_func, new_func_name)
        assert registered_func != orig_func
        assert registered_func.__name__ == new_func_name
