import re
import docstring_parser
import lazyllm
from lazyllm.tools import ToolManager
from lazyllm.tools.agent.toolsManager import MethodModuleTool
from lazyllm.tools.agent.toolsManager import (
    _gen_empty_func_str_from_parsed_docstring,
    _gen_func_from_str,
    _check_return_type_is_the_same,
    register,
)
from lazyllm.common import LazyLLMRegisterMetaClass
from typing import Literal, get_type_hints

def _gen_wrapped_moduletool(func):
    if 'tmp_tool' not in LazyLLMRegisterMetaClass.all_clses:
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
        func_str1 = 'def add1(v):\n    return v+1'
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
        def func(s: Literal['a', 'b', 'c']):
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
        doc2 = '''
        this is a function adding 5 to the value.

        Args:
            c (str): this is c's desc

        Returns:
            str: c itself
        '''

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
        doc3 = '''
        this is a function desc.

        Args:
            c (str): this is c's desc

        Returns:
            str: c itself
        '''

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


class MockSearchForTest:
    __public_apis__ = ['search', 'batch_search']

    def __init__(self, key: str = ''):
        self._key = key

    def search(self, query: str) -> str:
        '''Search the web for a query.

        Args:
            query (str): The search query string.

        Returns:
            str: The search result.
        '''
        return f'result:{query}'

    def batch_search(self, queries: str) -> str:
        '''Search the web for multiple queries.

        Args:
            queries (str): Comma-separated search queries.

        Returns:
            str: Combined search results.
        '''
        return ','.join(f'result:{q}' for q in queries.split(','))


class MockSearchWithClassKeySource:
    __public_apis__ = ['search']
    __key_source__ = lambda inst: inst._key  # noqa: E731

    def __init__(self, key: str = ''):
        self._key = key

    def search(self, query: str) -> str:
        '''Search the web.

        Args:
            query (str): The search query string.

        Returns:
            str: The search result.
        '''
        return f'result:{query}'


class MockSearchWithStrClassKeySource:
    __public_apis__ = ['search']
    __key_source__ = 'env.MOCK_SEARCH_STR_KEY'

    def __init__(self):
        pass

    def search(self, query: str) -> str:
        '''Search the web.

        Args:
            query (str): The search query string.

        Returns:
            str: The search result.
        '''
        return f'result:{query}'


class TestMethodModuleTool:
    def test_name_and_invocation_without_key(self):
        inst = MockSearchForTest()
        tool = MethodModuleTool(inst, 'search')
        assert tool.name == 'MockSearchForTest_search'
        assert tool.apply(query='hello') == 'result:hello'

    def test_should_skip_returns_false_when_no_key_source(self):
        tool = MethodModuleTool(MockSearchForTest(), 'search')
        assert tool.should_skip() is False

    def test_should_skip_callable_key_source_empty(self):
        inst = MockSearchForTest(key='')
        tool = MethodModuleTool(inst, 'search', lambda i: i._key)
        assert tool.should_skip() is True

    def test_should_skip_callable_key_source_present(self):
        inst = MockSearchForTest(key='valid-key')
        tool = MethodModuleTool(inst, 'search', lambda i: i._key)
        assert tool.should_skip() is False

    def test_should_skip_env_key_source(self, monkeypatch):
        inst = MockSearchForTest()
        tool = MethodModuleTool(inst, 'search', 'env.TEST_MMT_API_KEY')
        assert tool.should_skip() is True
        monkeypatch.setenv('TEST_MMT_API_KEY', 'env-value')
        assert tool.should_skip() is False

    def test_should_skip_multi_key_source_any_satisfies(self, monkeypatch):
        inst = MockSearchForTest(key='')
        tool = MethodModuleTool(inst, 'search', [lambda i: i._key, 'env.TEST_MULTI_KEY'])
        assert tool.should_skip() is True
        monkeypatch.setenv('TEST_MULTI_KEY', 'env-val')
        assert tool.should_skip() is False

    def test_should_skip_multi_key_source_first_satisfies(self):
        inst = MockSearchForTest(key='my-key')
        tool = MethodModuleTool(inst, 'search', [lambda i: i._key, 'env.NONEXISTENT_KEY_XYZ'])
        assert tool.should_skip() is False

    def test_should_skip_class_key_source_callable(self):
        inst = MockSearchWithClassKeySource(key='')
        tool = MethodModuleTool(inst, 'search')
        assert tool.should_skip() is True
        inst._key = 'valid'
        assert tool.should_skip() is False

    def test_should_skip_class_key_source_str(self, monkeypatch):
        inst = MockSearchWithStrClassKeySource()
        tool = MethodModuleTool(inst, 'search')
        assert tool.should_skip() is True
        monkeypatch.setenv('MOCK_SEARCH_STR_KEY', 'some-key')
        assert tool.should_skip() is False

    def test_explicit_key_source_overrides_class_key_source(self):
        inst = MockSearchWithClassKeySource(key='')
        tool = MethodModuleTool(inst, 'search', lambda i: 'override')
        assert tool.should_skip() is False

    def test_schema_contains_parameter_info(self):
        inst = MockSearchForTest()
        tool = MethodModuleTool(inst, 'search')
        schema = tool._params_schema
        assert schema is not None
        assert 'query' in schema.model_fields

    def test_tool_manager_loads_public_apis_as_tools(self):
        inst = MockSearchForTest(key='k')
        tm = ToolManager([(inst, lambda i: i._key)])
        names = {t.name for t in tm.all_tools}
        assert 'MockSearchForTest_search' in names
        assert 'MockSearchForTest_batch_search' in names

    def test_tool_manager_loads_bare_instance_with_class_key_source(self):
        inst = MockSearchWithClassKeySource(key='k')
        tm = ToolManager([inst])
        names = {t.name for t in tm.all_tools}
        assert 'MockSearchWithClassKeySource_search' in names
        assert len(tm.tools_description) == 1

    def test_tool_manager_bare_instance_hidden_when_no_key(self):
        inst = MockSearchWithClassKeySource(key='')
        tm = ToolManager([inst])
        assert len(tm.tools_description) == 0
        inst._key = 'valid'
        assert len(tm.tools_description) == 1

    def test_tools_description_filters_when_key_missing(self):
        inst = MockSearchForTest(key='')
        tm = ToolManager([(inst, lambda i: i._key)])
        assert len(tm.tools_description) == 0

    def test_tools_description_includes_when_key_present(self):
        inst = MockSearchForTest(key='')
        tm = ToolManager([(inst, lambda i: i._key)])
        assert len(tm.tools_description) == 0
        inst._key = 'valid-key'
        assert len(tm.tools_description) == 2

    def test_tool_manager_raises_without_public_apis(self):
        class NoPublicApis:
            pass

        try:
            ToolManager([(NoPublicApis(), 'env.K')])
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_parse_tool_call_returns_error_for_unavailable_tool(self):
        import json
        inst = MockSearchForTest(key='')
        tm = ToolManager([(inst, lambda i: i._key)])
        tc = {'function': {'name': 'MockSearchForTest_search',
                           'arguments': json.dumps({'query': 'hi'})}}
        tool, result = tm._parse_tool_call(tc)
        assert tool is None
        assert 'unavailable' in result.lower() or 'missing' in result.lower()
