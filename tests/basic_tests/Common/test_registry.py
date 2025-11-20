import lazyllm
from lazyllm.components import register as comp_register
from lazyllm.components.core import ComponentBase
from lazyllm.tools import fc_register

def orig_func(self):
    pass

class TestRegistry:

    def test_compoments_register(self):
        assert not hasattr(lazyllm, 'test1')
        comp_register.new_group('test1')
        assert hasattr(lazyllm, 'test1')

        @comp_register('test1')
        def test_add(a, b):
            return a + b

        @comp_register('test1')
        def TestSub(a, b):
            return a - b

        assert hasattr(lazyllm.test1, 'test_add')
        assert lazyllm.test1.test_add()(1, 2) == 3
        assert hasattr(lazyllm.test1, 'testsub')
        assert hasattr(lazyllm.test1, 'TestSub')
        assert lazyllm.test1.testsub()(3, 2) == 1

    def test_compoments_register_subgroup(self):
        assert not hasattr(lazyllm, 'test2')

        class LazyLLMTest2Base(ComponentBase):
            pass

        assert hasattr(lazyllm, 'test2')
        assert not hasattr(lazyllm.test2, 'test3')

        class LazyLLMTest3Base(LazyLLMTest2Base):
            pass

        assert hasattr(lazyllm.test2, 'test3')

        @comp_register('test2.test3')
        def test_add(a, b):
            return a + b + 1

        assert lazyllm.test2.test3.test_add()(1, 2) == 4

    def test_compoments_register_with_default_group(self):
        assert not hasattr(lazyllm, 'test3')
        comp_register.new_group('test3')
        comp_register._default_group = 'test3'
        assert not hasattr(lazyllm.test3, 'test_square')

        @comp_register
        def test_square(x):
            return x * x

        assert lazyllm.test3.test_square()(4) == 16

    def test_capital_letter(self):
        class LazyLLMTest5Base(ComponentBase):
            pass

        assert hasattr(lazyllm, 'test5')
        assert hasattr(lazyllm, 'Test5')

        class a(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'a')
        assert hasattr(lazyllm.test5, 'A')

        class B(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'b')
        assert hasattr(lazyllm.test5, 'B')

        class cTest5(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'c')
        assert hasattr(lazyllm.test5, 'C')
        assert hasattr(lazyllm.test5, 'ctest5')
        assert hasattr(lazyllm.test5, 'CTest5')

        class DTest5(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'd')
        assert hasattr(lazyllm.test5, 'D')
        assert hasattr(lazyllm.test5, 'dtest5')
        assert hasattr(lazyllm.test5, 'DTest5')

    def test_register(self):
        registered_func = fc_register('tool')(orig_func)
        assert registered_func == orig_func

    def test_register_with_new_func_name(self):
        new_func_name = 'another_func_name'
        registered_func = fc_register('tool')(orig_func, new_func_name)
        assert registered_func != orig_func
        assert registered_func.__name__ == new_func_name
