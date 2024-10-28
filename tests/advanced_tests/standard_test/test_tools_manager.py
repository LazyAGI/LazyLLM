import re
import docstring_parser
import lazyllm
from lazyllm.tools import ToolManager
from lazyllm.tools.agent.toolsManager import (
    _gen_empty_func_str_from_parsed_docstring,
    _gen_func_from_str,
    _check_return_type_is_the_same,
    register,
)
from lazyllm.common import LazyLLMRegisterMetaClass
from typing import Literal, get_type_hints

def _gen_wrapped_moduletool(func):
    if "tmp_tool" not in LazyLLMRegisterMetaClass.all_clses:
        register.new_group('tmp_tool')
    register('tmp_tool')(func)
    wrapped_module = getattr(lazyllm.tmp_tool, func.__name__)()
    lazyllm.tmp_tool.remove(func.__name__)
    return wrapped_module

class TestToolManager:
    def test_gen_empty_func_str_from_parsed_docstring(self):
        var_args_doc = '''
        this is a function with *args and **kwargs.

        Args:
            a (int): this is an required integer.
            b (Literal['foo', 'bar', 'baz']): this is a str with candidate values.
            c (List[str]): this is a string list.

        Returns:
            str: returns a string
        '''

        parsed_doc = docstring_parser.parse(var_args_doc)
        func_str = _gen_empty_func_str_from_parsed_docstring(parsed_doc)
        expected_pattern = "def f[0-9]+\(a:int,b:Literal\['foo', 'bar', 'baz'\],c:List\[str\],\)->str:\n    pass"  # noqa W605
        res = re.match(expected_pattern, func_str)
        assert res.span()
        assert (res.span()[1] - res.span()[0]) == len(func_str)

    def test_gen_func_from_str(self):
        func_str1 = "def add1(v):\n    return v+1"
        func_doc1 = '''
        this is a function returning value + 1.

        Args:
            v (int): value
        '''
        func = _gen_func_from_str(func_str1, func_doc1)
        value = 5
        assert func(value) == (value + 1)
        assert func.__doc__ == func_doc1

    def test_enum_func(self):
        def func(s: Literal["a", "b", "c"]):
            '''
            whatever
            '''
            pass

        doc = '''
        this is a function with Literal.

        Args:
            s (Literal['a', 'b', 'c']): a string
        '''
        parsed_doc = docstring_parser.parse(doc)
        tool = _gen_wrapped_moduletool(func)
        args = ToolManager._gen_args_info_from_moduletool_and_docstring(tool, parsed_doc)
        info_of_s = args['s']
        assert info_of_s['enum'] == ['a', 'b', 'c']

    def test_check_return_type_is_the_same(self):
        def add5(v: int) -> int:
            '''
            this is a function adding 5 to the value.

            Args:
                v (int): this is v's desc in function, different from that in doc

            Returns:
                int: value+5
            '''
            return v + 5

        add5_doc1 = '''
        this is a function adding 5 to the value.

        Args:
            v (int): this is v's desc

        Returns:
            int: value+5
        '''

        func_str = 'def add5(v:int) -> int:\n    return v+5'
        func_from_doc = _gen_func_from_str(func_str, add5_doc1)
        tool = _gen_wrapped_moduletool(add5)
        try:
            value = 123
            _check_return_type_is_the_same(
                get_type_hints(func_from_doc, globals(), locals()),
                get_type_hints(tool.__call__, globals(), locals()))
            value = 456
        except Exception:
            value = 789
        finally:
            assert value == 456

        # ----- #

        str_without_arg_type = 'def identity(c) -> str:\n    return c'
        doc2 = """
        this is a function adding 5 to the value.

        Args:
            c (str): this is c's desc

        Returns:
            str: c itself
        """

        func_from_doc2 = _gen_func_from_str(str_without_arg_type, doc2)
        try:
            value = 123
            _check_return_type_is_the_same(
                get_type_hints(func_from_doc2, globals(), locals()),
                get_type_hints(tool.__call__, globals(), locals()))
            value = 456
        except Exception:
            value = 789
        finally:
            assert value == 456

        # ----- #

        str_without_return_type = 'def identity3(c:str):\n    return c'
        doc3 = """
        this is a function desc.

        Args:
            c (str): this is c's desc

        Returns:
            str: c itself
        """

        func_from_doc3 = _gen_func_from_str(str_without_return_type, doc3)
        try:
            value = 123
            _check_return_type_is_the_same(
                get_type_hints(func_from_doc3, globals(), locals()),
                get_type_hints(tool.__call__, globals(), locals()))
            value = 456
        except Exception:
            value = 789
        finally:
            assert value == 456

    def test_invalid_typing(self):
        invalid_doc1 = '''
        this is an doc containing invalid types

        Args:
            v (Str): this is v desc

        Returns:
            int: return value
        '''

        parsed_docstring = docstring_parser.parse(invalid_doc1)
        func_str = _gen_empty_func_str_from_parsed_docstring(parsed_docstring)
        try:
            test_value = 123
            _gen_func_from_str(func_str, invalid_doc1)
            test_value = 456
        except Exception:
            test_value = 789
        finally:
            assert test_value == 789

    def test_case_sensitivity_of_generic_type(self):
        invalid_doc1 = '''
        this is an doc containing invalid types

        Args:
            v (union[str, Any]): this is v desc

        Returns:
            int: return value
        '''

        parsed_docstring = docstring_parser.parse(invalid_doc1)
        func_str = _gen_empty_func_str_from_parsed_docstring(parsed_docstring)
        try:
            test_value = 111
            _gen_func_from_str(func_str, invalid_doc1)
            test_value = 222
        except Exception:
            test_value = 333
        finally:
            assert test_value == 333
