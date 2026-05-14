import lazyllm
from lazyllm.tools import ToolManager
from lazyllm.tools.agent.toolsManager import InstanceToolGroup, _gen_args_info_from_moduletool_and_docstring
from lazyllm.tools.agent.toolsManager import register
from lazyllm.common import LazyLLMRegisterMetaClass
from typing import Literal

def _gen_wrapped_moduletool(func):
    if 'tmp_tool' not in LazyLLMRegisterMetaClass.all_clses:
        register.new_group('tmp_tool')
    register('tmp_tool')(func)
    wrapped_module = getattr(lazyllm.tmp_tool, func.__name__)()
    lazyllm.tmp_tool.remove(func.__name__)
    return wrapped_module

class TestToolManager:
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
        import docstring_parser
        parsed_doc = docstring_parser.parse(doc)
        tool = _gen_wrapped_moduletool(func)
        args = _gen_args_info_from_moduletool_and_docstring(tool, parsed_doc)
        info_of_s = args['s']
        assert info_of_s['enum'] == ['a', 'b', 'c']

    def test_check_return_type_mismatch_raises(self):
        def mismatched(v: int) -> str:
            '''
            A function with mismatched return type.

            Args:
                v (int): input value

            Returns:
                int: should be int but annotation says str
            '''
            return str(v)

        try:
            _gen_wrapped_moduletool(mismatched)
            raised = False
        except TypeError:
            raised = True
        assert raised

    def test_unsafe_type_expression_in_docstring_raises(self):
        def dangerous_tool(v):
            '''
            A tool with a dangerous type expression.

            Args:
                v (exec('pass')): dangerous type
            '''
            return v

        try:
            _gen_wrapped_moduletool(dangerous_tool)
            raised = False
        except Exception:
            raised = True
        assert raised

    def test_invalid_typing_in_docstring_raises(self):
        def bad_type(v):
            '''
            A function with invalid type in docstring.

            Args:
                v (Str): invalid type name
            '''
            return v

        try:
            _gen_wrapped_moduletool(bad_type)
            raised = False
        except Exception:
            raised = True
        assert raised

    def test_case_sensitivity_of_generic_type(self):
        def bad_union(v):
            '''
            A function with lowercase union type.

            Args:
                v (union[str, Any]): invalid lowercase union
            '''
            return v

        try:
            _gen_wrapped_moduletool(bad_union)
            raised = False
        except Exception:
            raised = True
        assert raised


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


class TestInstanceToolGroup:
    def test_name_and_invocation_without_key(self):
        inst = MockSearchForTest()
        grp = InstanceToolGroup(inst)
        tool = grp._tools['MockSearchForTest_search']
        assert tool.name == 'MockSearchForTest_search'
        assert tool.apply(query='hello') == 'result:hello'

    def test_should_skip_returns_false_when_no_key_source(self):
        grp = InstanceToolGroup(MockSearchForTest())
        assert grp.should_skip() is False

    def test_should_skip_callable_key_source_empty(self):
        inst = MockSearchForTest(key='')
        grp = InstanceToolGroup(inst, lambda i: i._key)
        assert grp.should_skip() is True

    def test_should_skip_callable_key_source_present(self):
        inst = MockSearchForTest(key='valid-key')
        grp = InstanceToolGroup(inst, lambda i: i._key)
        assert grp.should_skip() is False

    def test_should_skip_env_key_source(self, monkeypatch):
        inst = MockSearchForTest()
        grp = InstanceToolGroup(inst, 'env.TEST_MMT_API_KEY')
        assert grp.should_skip() is True
        monkeypatch.setenv('TEST_MMT_API_KEY', 'env-value')
        assert grp.should_skip() is False

    def test_should_skip_multi_key_source_any_satisfies(self, monkeypatch):
        inst = MockSearchForTest(key='')
        grp = InstanceToolGroup(inst, [lambda i: i._key, 'env.TEST_MULTI_KEY'])
        assert grp.should_skip() is True
        monkeypatch.setenv('TEST_MULTI_KEY', 'env-val')
        assert grp.should_skip() is False

    def test_should_skip_multi_key_source_first_satisfies(self):
        inst = MockSearchForTest(key='my-key')
        grp = InstanceToolGroup(inst, [lambda i: i._key, 'env.NONEXISTENT_KEY_XYZ'])
        assert grp.should_skip() is False

    def test_should_skip_class_key_source_callable(self):
        inst = MockSearchWithClassKeySource(key='')
        grp = InstanceToolGroup(inst)
        assert grp.should_skip() is True
        inst._key = 'valid'
        assert grp.should_skip() is False

    def test_should_skip_class_key_source_str(self, monkeypatch):
        inst = MockSearchWithStrClassKeySource()
        grp = InstanceToolGroup(inst)
        assert grp.should_skip() is True
        monkeypatch.setenv('MOCK_SEARCH_STR_KEY', 'some-key')
        assert grp.should_skip() is False

    def test_explicit_key_source_overrides_class_key_source(self):
        inst = MockSearchWithClassKeySource(key='')
        grp = InstanceToolGroup(inst, lambda i: 'override')
        assert grp.should_skip() is False

    def test_schema_contains_parameter_info(self):
        inst = MockSearchForTest()
        grp = InstanceToolGroup(inst)
        tool = grp._tools['MockSearchForTest_search']
        assert tool._params_schema is not None
        assert 'query' in tool._params_schema.model_fields

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

    def test_unavailable_tool_hidden_from_description(self):
        inst = MockSearchForTest(key='')
        tm = ToolManager([(inst, lambda i: i._key)])
        assert len(tm.tools_description) == 0
        inst._key = 'valid'
        assert len(tm.tools_description) == len(inst.__public_apis__)
