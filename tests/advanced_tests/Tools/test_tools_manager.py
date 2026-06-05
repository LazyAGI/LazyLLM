import lazyllm
from lazyllm.tools import ToolManager
from lazyllm.tools.agent.toolsManager import (
    InstanceToolGroup, ToolGroup, _build_tool_from_element, _gen_args_info_from_moduletool_and_docstring, register,
)
from lazyllm.common import LazyLLMRegisterMetaClass
from lazyllm import locals as lazyllm_locals, init_session
from typing import Literal

def _gen_wrapped_moduletool(func):
    if 'tmp_tool' not in LazyLLMRegisterMetaClass.all_clses:
        register.new_group('tmp_tool')
    register('tmp_tool')(func)
    wrapped_module = lazyllm.tmp_tool.resolve(func.__name__)()
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

    def test_tmp_tool_name_can_match_dict_method(self):
        def copy(path1: str, path2: str) -> str:
            '''
            Copy a file.

            Args:
                path1 (str): Source path.
                path2 (str): Target path.

            Returns:
                str: Copy result.
            '''
            return f'{path1}->{path2}'

        tool = _build_tool_from_element(copy)
        assert not isinstance(tool, dict)
        assert tool.name == 'copy'


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
        # lazy mode: only the gateway tool is exposed initially
        assert len(tm.tools_description) == 1
        assert tm.tools_description[0]['function']['name'] == 'get_MockSearchForTest_methods'

    def test_tool_manager_raises_without_public_apis(self):
        class NoPublicApis:
            pass

        try:
            ToolManager([(NoPublicApis(), 'env.K')])
            raised = False
        except (ValueError, TypeError):
            raised = True
        assert raised

    def test_unavailable_tool_hidden_from_description(self):
        inst = MockSearchForTest(key='')
        tm = ToolManager([(inst, lambda i: i._key)])
        assert len(tm.tools_description) == 0
        inst._key = 'valid'
        # lazy mode: only the gateway tool is exposed initially (1 gateway, not len(public_apis))
        assert len(tm.tools_description) == 1
        assert tm.tools_description[0]['function']['name'] == 'get_MockSearchForTest_methods'


def _make_tool_func(name: str, param: str = 'x'):
    def fn(**kwargs): return kwargs.get(param, '')
    fn.__name__ = name
    fn.__doc__ = (
        f'{name} tool.\n\nArgs:\n    {param} (str): input.\n\nReturns:\n    str: output.\n'
    )
    import inspect
    fn.__annotations__ = {param: str, 'return': str}
    fn.__signature__ = inspect.Signature([inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                            annotation=str)])
    return fn


class TestToolGroup:
    def setup_method(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}

    def _tool(self, name: str):
        return _make_tool_func(name)

    def test_eager_mode_exposes_all_tools_directly(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', lazy=False, tools=[t1, t2])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'alpha' in names
        assert 'beta' in names
        assert 'get_grp_methods' not in names

    def test_lazy_mode_exposes_only_gateway_initially(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', lazy=True, tools=[t1, t2])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert names == ['get_grp_methods']
        assert 'alpha' not in names
        assert 'beta' not in names

    def test_lazy_mode_default_is_lazy(self):
        t1 = self._tool('alpha')
        tm = ToolManager([dict(name='grp', desc='Group', tools=[t1])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert names == ['get_grp_methods']

    def test_gateway_tool_registered_in_tool_call(self):
        t1 = self._tool('alpha')
        tm = ToolManager([dict(name='grp', desc='Group', tools=[t1])])
        assert 'get_grp_methods' in tm._tool_call
        assert 'alpha' in tm._tool_call

    def test_gateway_tool_activation_updates_tools_description(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', tools=[t1, t2])])
        assert [d['function']['name'] for d in tm.tools_description] == ['get_grp_methods']

        gateway = tm._tool_call['get_grp_methods']
        result = gateway({})
        assert 'alpha' in result
        assert 'beta' in result

        names = [d['function']['name'] for d in tm.tools_description]
        assert 'get_grp_methods' not in names
        assert 'alpha' in names
        assert 'beta' in names

    def test_gateway_tool_result_format(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', tools=[t1, t2])])
        gateway = tm._tool_call['get_grp_methods']
        result = gateway({})
        assert result.startswith('Activated tool group "grp".')
        assert 'alpha' in result
        assert 'beta' in result

    def test_multilevel_lazy_outer_exposes_outer_gateway(self):
        t1, t2, t3 = self._tool('a'), self._tool('b'), self._tool('c')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', tools=[t2, t3]),
        ])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert names == ['get_outer_methods']
        assert 'get_inner_methods' not in names

    def test_multilevel_lazy_activating_outer_exposes_inner_gateway(self):
        t1, t2, t3 = self._tool('a'), self._tool('b'), self._tool('c')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', tools=[t2, t3]),
        ])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'a' not in names
        assert 'b' not in names
        assert 'c' not in names
        assert 'get_inner_methods' not in names
        tm._tool_call['get_outer_methods']({})
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'get_outer_methods' not in names
        assert 'a' in names
        assert 'get_inner_methods' in names
        assert 'b' not in names
        assert 'c' not in names

    def test_multilevel_lazy_activating_inner_exposes_leaf_tools(self):
        t1, t2, t3 = self._tool('a'), self._tool('b'), self._tool('c')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', tools=[t2, t3]),
        ])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'a' not in names
        assert 'b' not in names
        assert 'c' not in names
        assert 'get_inner_methods' not in names
        tm._tool_call['get_outer_methods']({})
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'b' not in names
        assert 'c' not in names
        tm._tool_call['get_inner_methods']({})
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'get_outer_methods' not in names
        assert 'get_inner_methods' not in names
        assert 'a' in names
        assert 'b' in names
        assert 'c' in names

    def test_eager_inner_group_exposes_leaf_tools_after_outer_activation(self):
        t1, t2, t3 = self._tool('a'), self._tool('b'), self._tool('c')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', lazy=False, tools=[t2, t3]),
        ])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'a' not in names
        assert 'b' not in names
        assert 'c' not in names
        assert 'get_inner_methods' not in names
        tm._tool_call['get_outer_methods']({})
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'a' in names
        assert 'b' in names
        assert 'c' in names
        assert 'get_inner_methods' not in names

    def test_dict_missing_name_raises(self):
        t1 = self._tool('alpha')
        try:
            ToolManager([dict(desc='Group', tools=[t1])])
            raised = False
        except AssertionError:
            raised = True
        assert raised

    def test_dict_missing_tools_raises(self):
        try:
            ToolManager([dict(name='grp', desc='Group')])
            raised = False
        except AssertionError:
            raised = True
        assert raised

    def test_toolgroup_with_callable_children(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        grp = ToolGroup(tools=[t1, t2], name='grp', desc='Group')
        flat = grp.get_flat_tools()
        # default prefix=True, so tool names are prefixed with group name
        assert 'grp_alpha' in flat
        assert 'grp_beta' in flat

    def test_toolgroup_get_description_lazy(self):
        t1 = self._tool('alpha')
        grp = ToolGroup(tools=[t1], name='grp', desc='My group', lazy=True)
        descs = grp.get_description()
        assert len(descs) == 1
        assert descs[0]['function']['name'] == 'get_grp_methods'
        # desc = "Get available methods of grp. My group"
        assert 'Get available methods of grp' in descs[0]['function']['description']
        assert 'My group' in descs[0]['function']['description']

    def test_toolgroup_get_description_eager(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        grp = ToolGroup(tools=[t1, t2], name='grp', desc='Group', lazy=False)
        descs = grp.get_description()
        names = [d['function']['name'] for d in descs]
        # default prefix=True, so tool names are prefixed with group name
        assert 'grp_alpha' in names
        assert 'grp_beta' in names

    def test_toolgroup_isolation_across_sessions(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', tools=[t1, t2])])

        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
        assert [d['function']['name'] for d in tm.tools_description] == ['get_grp_methods']

        tm._tool_call['get_grp_methods']({})
        assert 'alpha' in [d['function']['name'] for d in tm.tools_description]

        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
        assert [d['function']['name'] for d in tm.tools_description] == ['get_grp_methods']

    def test_mixed_flat_and_group_tools(self):
        t1, t2, t3 = self._tool('flat_tool'), self._tool('grp_a'), self._tool('grp_b')
        tm = ToolManager([t1, dict(name='grp', desc='Group', tools=[t2, t3])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'flat_tool' in names
        assert 'get_grp_methods' in names
        assert 'grp_a' not in names
        assert 'grp_b' not in names

    def test_prefix_true_adds_group_name_to_tool_names(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', lazy=False, prefix=True, tools=[t1, t2])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'grp_alpha' in names
        assert 'grp_beta' in names
        assert 'alpha' not in names
        assert 'beta' not in names

    def test_prefix_false_keeps_original_tool_names(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        tm = ToolManager([dict(name='grp', desc='Group', lazy=False, prefix=False, tools=[t1, t2])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'alpha' in names
        assert 'beta' in names
        assert 'grp_alpha' not in names

    def test_prefix_false_tool_callable_by_original_name(self):
        t1 = self._tool('alpha')
        tm = ToolManager([dict(name='grp', desc='Group', lazy=False, prefix=False, tools=[t1])])
        assert 'alpha' in tm._tool_call
        assert 'grp_alpha' not in tm._tool_call

    def test_gateway_available_tools_lists_direct_children_only(self):
        # 嵌套时 gateway 的 Available tools 只列直接子工具名，不递归展开子 group 的叶子
        t1 = self._tool('tool_aaa')
        t2, t3 = self._tool('tool_bbb'), self._tool('tool_ccc')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', tools=[t2, t3]),
        ])])
        result = tm._tool_call['get_outer_methods']({})
        assert 'tool_aaa' in result
        assert 'get_inner_methods' in result
        assert 'tool_bbb' not in result
        assert 'tool_ccc' not in result

    def test_eager_inner_leaf_names_included_in_outer_gateway(self):
        # 非 lazy 子 group 的叶子名直接出现在父 gateway 的 Available tools 里
        t1, t2, t3 = self._tool('a'), self._tool('b'), self._tool('c')
        tm = ToolManager([dict(name='outer', desc='Outer', tools=[
            t1,
            dict(name='inner', desc='Inner', lazy=False, tools=[t2, t3]),
        ])])
        result = tm._tool_call['get_outer_methods']({})
        assert 'a' in result
        assert 'b' in result
        assert 'c' in result
        assert 'get_inner_methods' not in result

    def test_prefix_propagates_to_nested_group_leaf_names(self):
        t1 = self._tool('leaf')
        tm = ToolManager([dict(name='outer', desc='Outer', lazy=False, prefix=True, tools=[
            dict(name='inner', desc='Inner', lazy=False, prefix=True, tools=[t1]),
        ])])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'outer_inner_leaf' in names


class TestToolGroupWrapper:
    def setup_method(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}

    def _tool(self, name: str):
        return _make_tool_func(name)

    def test_wrapper_with_callable_tool_and_callable_key_present(self):
        t = self._tool('my_tool')
        tm = ToolManager([(t, lambda: 'valid-key')])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'my_tool' in names

    def test_wrapper_with_callable_tool_and_callable_key_missing(self):
        t = self._tool('my_tool')
        tm = ToolManager([(t, lambda: '')])
        assert len(tm.tools_description) == 0

    def test_wrapper_with_callable_key_with_instance_arg(self):
        t = self._tool('my_tool')
        called_with = []

        def key_fn(inst):
            called_with.append(inst)
            return 'valid'

        tm = ToolManager([(t, key_fn)])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'my_tool' in names
        assert called_with == [None]

    def test_wrapper_with_env_key_source(self, monkeypatch):
        t = self._tool('my_tool')
        tm = ToolManager([(t, 'env.TEST_WRAPPER_KEY')])
        assert len(tm.tools_description) == 0
        monkeypatch.setenv('TEST_WRAPPER_KEY', 'some-val')
        assert len(tm.tools_description) == 1
        assert tm.tools_description[0]['function']['name'] == 'my_tool'

    def test_wrapper_with_multi_key_source_any_satisfies(self, monkeypatch):
        t = self._tool('my_tool')
        tm = ToolManager([(t, [lambda: '', 'env.TEST_WRAPPER_MULTI_KEY'])])
        assert len(tm.tools_description) == 0
        monkeypatch.setenv('TEST_WRAPPER_MULTI_KEY', 'val')
        assert len(tm.tools_description) == 1

    def test_wrapper_with_toolgroup_inner(self):
        t1, t2 = self._tool('alpha'), self._tool('beta')
        grp = ToolGroup(tools=[t1, t2], name='grp', desc='Group', lazy=False, prefix=False)
        tm = ToolManager([(grp, lambda: 'key')])
        names = [d['function']['name'] for d in tm.tools_description]
        assert 'alpha' in names
        assert 'beta' in names

    def test_wrapper_with_toolgroup_inner_hidden_when_no_key(self):
        t1 = self._tool('alpha')
        grp = ToolGroup(tools=[t1], name='grp', desc='Group', lazy=False, prefix=False)
        tm = ToolManager([(grp, lambda: '')])
        assert len(tm.tools_description) == 0

    def test_wrapper_get_flat_tools_contains_inner_tools(self):
        t = self._tool('my_tool')
        tm = ToolManager([(t, lambda: 'key')])
        assert 'my_tool' in tm._tool_call

    def test_wrapper_no_key_source_never_skips(self):
        from lazyllm.tools.agent.toolsManager import ToolGroupWrapper
        t = self._tool('my_tool')
        wrapper = ToolGroupWrapper(t, None)
        assert wrapper.should_skip() is False


class TestSkipMixinNormalizeSource:
    def test_no_arg_callable_wrapped(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        called = []

        def no_arg():
            called.append(1)
            return 'key'

        mixin = SkipMixin(no_arg)
        assert not mixin.should_skip()
        assert called

    def test_one_arg_callable_passed_none(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        received = []

        def one_arg(inst):
            received.append(inst)
            return 'key'

        mixin = SkipMixin(one_arg)
        assert not mixin.should_skip()
        assert received == [None]

    def test_list_mixed_callable_normalized(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin

        def no_arg(): return ''
        def one_arg(inst): return 'key'  # noqa: E731

        mixin = SkipMixin([no_arg, one_arg])
        assert not mixin.should_skip()

    def test_defaultonly_arg_callable_treated_as_no_arg(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin

        def optional_arg(inst=None): return 'key'

        mixin = SkipMixin(optional_arg)
        assert not mixin.should_skip()


class TestShouldSkipTransitions:
    def setup_method(self):
        init_session()
        lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
        # register test-only config keys once (idempotent via _supported_configs set)
        lazyllm.globals.config.add('test_skip_api_key', str, None, 'TEST_SKIP_API_KEY',
                                   description='test key for should_skip transitions')
        lazyllm.globals.config.add('test_skip_dot_key', str, None, 'TEST_SKIP_DOT_KEY',
                                   description='test key for globals.config.xxx format')
        lazyllm.globals.config.add('mock_fs_token', str, None, 'MOCK_FS_TOKEN',
                                   description='test key for CredentialMixin skip transitions')
        lazyllm.globals.config.add('test_wrapper_cfg_key', str, None, 'TEST_WRAPPER_CFG_KEY',
                                   description='test key for ToolGroupWrapper globals.config')

    # ------------------------------------------------------------------ #
    # InstanceToolGroup — globals.config string key_source (bare key)
    # ------------------------------------------------------------------ #
    def test_instance_group_globals_config_key_skip_then_available(self):
        inst = MockSearchForTest()
        grp = InstanceToolGroup(inst, 'test_skip_api_key')
        assert grp.should_skip() is True
        lazyllm.globals.config['test_skip_api_key'] = 'secret'
        assert grp.should_skip() is False
        lazyllm.globals.config['test_skip_api_key'] = None

    def test_instance_group_globals_config_dot_prefix_skip_then_available(self):
        inst = MockSearchForTest()
        grp = InstanceToolGroup(inst, 'globals.config.test_skip_dot_key')
        assert grp.should_skip() is True
        lazyllm.globals.config['test_skip_dot_key'] = 'token-xyz'
        assert grp.should_skip() is False
        lazyllm.globals.config['test_skip_dot_key'] = None

    # ------------------------------------------------------------------ #
    # InstanceToolGroup — CredentialMixin auto key_source via __key_source__
    # ------------------------------------------------------------------ #
    def test_instance_group_credential_mixin_skip_then_available(self):
        from lazyllm.common import CredentialMixin
        from lazyllm.common.auth import Credential

        class MockFS(CredentialMixin):
            __public_apis__ = ['read']

            def __init__(self):
                self.__init_credential__(Credential(kind='dynamic'))

            def _resolve_dynamic_token(self) -> str:
                try:
                    return lazyllm.globals.config['mock_fs_token'] or ''
                except Exception:
                    return ''

            def read(self, path: str) -> str:
                '''Read a file.

                Args:
                    path (str): File path.

                Returns:
                    str: File content.
                '''
                return ''

        inst = MockFS()
        grp = InstanceToolGroup(inst)
        # no token yet → should skip
        assert grp.should_skip() is True
        # inject token via globals.config
        lazyllm.globals.config['mock_fs_token'] = 'user-token-abc'
        assert grp.should_skip() is False
        lazyllm.globals.config['mock_fs_token'] = None

    def test_required_search_without_key_skips_then_available_with_dynamic_auth(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        from lazyllm.tools.tools.search import BingSearch

        old_auth = lazyllm.globals.config['dynamic_tool_auth']
        lazyllm.globals.config['dynamic_tool_auth'] = {}
        try:
            inst = BingSearch()
            mixin = SkipMixin(type(inst).__key_source__)
            mixin._instance = inst
            assert mixin.should_skip() is True

            lazyllm.globals.config['dynamic_tool_auth'] = {'bing': 'bing-token'}
            assert mixin.should_skip() is False
        finally:
            lazyllm.globals.config['dynamic_tool_auth'] = old_auth

    def test_search_skip_auth_override_keeps_tool_available(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        from lazyllm.tools.tools.search import ArxivSearch, WikipediaSearch

        for inst in (ArxivSearch(skip_auth=True), WikipediaSearch(skip_auth=True)):
            mixin = SkipMixin(type(inst).__key_source__)
            mixin._instance = inst
            assert mixin.should_skip() is False

    def test_search_without_key_skips_by_default(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        from lazyllm.tools.tools.search import ArxivSearch

        inst = ArxivSearch()
        mixin = SkipMixin(type(inst).__key_source__)
        mixin._instance = inst
        assert mixin.should_skip() is True

    # ------------------------------------------------------------------ #
    # ToolGroupWrapper — callable key_source, mutate state
    # ------------------------------------------------------------------ #
    def test_wrapper_callable_skip_then_available_via_state(self):
        from lazyllm.tools.agent.toolsManager import ToolGroupWrapper
        t = _make_tool_func('state_tool')
        state = {'key': ''}
        wrapper = ToolGroupWrapper(t, lambda _inst: state['key'])
        assert wrapper.should_skip() is True
        state['key'] = 'valid-key'
        assert wrapper.should_skip() is False

    # ------------------------------------------------------------------ #
    # ToolGroupWrapper — globals.config string key_source
    # ------------------------------------------------------------------ #
    def test_wrapper_globals_config_skip_then_available(self):
        from lazyllm.tools.agent.toolsManager import ToolGroupWrapper
        t = _make_tool_func('cfg_tool')
        wrapper = ToolGroupWrapper(t, 'test_wrapper_cfg_key')
        assert wrapper.should_skip() is True
        lazyllm.globals.config['test_wrapper_cfg_key'] = 'present'
        assert wrapper.should_skip() is False
        lazyllm.globals.config['test_wrapper_cfg_key'] = None

    # ------------------------------------------------------------------ #
    # dict tool group with key_source — via ToolManager
    # ------------------------------------------------------------------ #
    def test_dict_group_key_source_skip_then_available(self):
        t1, t2 = _make_tool_func('da'), _make_tool_func('db')
        state = {'key': ''}
        tm = ToolManager([dict(name='dgrp', desc='D group', tools=[t1, t2],
                               key_source=lambda _inst: state['key'])])
        assert len(tm.tools_description) == 0
        state['key'] = 'unlocked'
        # lazy mode: gateway tool becomes visible
        assert len(tm.tools_description) == 1
        assert tm.tools_description[0]['function']['name'] == 'get_dgrp_methods'

    # ------------------------------------------------------------------ #
    # SkipMixin — multi-source: both empty → skip; one becomes non-empty → available
    # ------------------------------------------------------------------ #
    def test_skip_mixin_multi_source_both_empty_then_one_fills(self):
        from lazyllm.tools.agent.toolsManager import SkipMixin
        state = {'a': '', 'b': ''}
        mixin = SkipMixin([lambda _: state['a'], lambda _: state['b']])
        assert mixin.should_skip() is True
        state['b'] = 'filled'
        assert mixin.should_skip() is False
