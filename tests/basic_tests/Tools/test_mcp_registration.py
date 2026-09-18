import copy
import re

import pytest
from types import SimpleNamespace

from lazyllm.tools import get_tool_runtime_metadata
from lazyllm.tools.agent.toolsManager import ToolManager, fc_register
from lazyllm.tools.mcp import MCPClient
from lazyllm.tools.mcp.tool_adaptor import generate_lazyllm_tool
from mcp.types import CallToolResult, TextContent


def make_tool(server_id='', name='search'):
    client = MCPClient('https://example.test/mcp', server_id=server_id)

    async def call_tool(tool_name, arguments):
        return CallToolResult(content=[TextContent(type='text', text=f'{server_id}:{tool_name}')])

    client.call_tool = call_tool
    return generate_lazyllm_tool(client, SimpleNamespace(
        name=name, description='Search documents.', inputSchema={'type': 'object', 'properties': {}}))


def test_combined_registration_preserves_cached_tools_routing_and_identity():
    tools = [make_tool('a'), make_tool('b')]
    assert get_tool_runtime_metadata(tools[0]).tool_origin == 'a'
    manager = ToolManager(tools)
    catalog = manager.atomic_tool_catalog()
    expected = set(catalog)
    assert len(expected) == 2
    assert all(name.startswith('search_mcp_') for name in expected)
    assert len({entry['identity'] for entry in catalog.values()}) == 2
    assert {entry['origin'] for entry in catalog.values()} == {'a', 'b'}
    for name, entry in catalog.items():
        assert f"{entry['origin']}:search" in manager.tools_info[name]({})
    assert [tool.__name__ for tool in tools] == ['search', 'search']
    assert set(ToolManager(list(reversed(tools))).atomic_tool_catalog()) == expected
    assert set(ToolManager([tools[0]]).atomic_tool_catalog()) <= expected
    assert {d['function']['name'] for d in manager.tools_description} == expected


def test_legacy_no_id_and_tuple_registration_keep_names():
    legacy = make_tool()
    identified = make_tool('a')
    manager = ToolManager([legacy, (identified, lambda: True)])
    catalog = manager.atomic_tool_catalog()
    assert catalog['search']['origin'] == ''
    assert catalog['search']['identity'].startswith('temporary:')
    assert len(catalog) == 2
    assert list(ToolManager([legacy]).atomic_tool_catalog()) == ['search']


def test_registered_local_name_reserves_alias():
    @fc_register('tool')
    def mcp_registration_search() -> str:
        '''Search local documents.'''
        return 'local'

    tool = make_tool('a', 'mcp_registration_search')
    manager = ToolManager(['mcp_registration_search', tool])
    catalog = manager.atomic_tool_catalog()
    assert len(catalog) == 2
    assert catalog['mcp_registration_search']['source'] == 'local'
    assert next(name for name, entry in catalog.items() if entry['source'] == 'mcp').startswith(
        'mcp_registration_search_mcp_')


def test_wrapped_registered_tool_constructed_once_and_keeps_key_source(monkeypatch):
    import lazyllm

    @fc_register('tool')
    def wrapped_mcp_registration_search() -> str:
        '''Search local documents.'''
        return 'local'

    tool_class = lazyllm.tool.wrapped_mcp_registration_search
    original_init = tool_class.__init__
    instances = []

    def count_init(self, *args, **kwargs):
        instances.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(tool_class, '__init__', count_init)
    enabled = [True]
    manager = ToolManager([('wrapped_mcp_registration_search', lambda: enabled[0]),
                           make_tool('a', 'wrapped_mcp_registration_search')])
    assert len(instances) == 1
    assert len(manager.atomic_tool_catalog()) == 2
    enabled[0] = False
    catalog = manager.atomic_tool_catalog()
    assert len(catalog) == 1
    assert next(iter(catalog.values()))['source'] == 'mcp'


class RetrievalStateStore:
    def __init__(self):
        self.state = {}

    def read(self):
        return copy.deepcopy(self.state)

    def update(self, update):
        self.state = update(self.read())
        return self.read()


def test_loaded_mcp_survives_catalog_changes_without_exposing_local_tool():
    def search() -> str:
        '''Search local documents.'''
        return 'local'

    state_store = RetrievalStateStore()
    a, b = make_tool('a'), make_tool('b')
    original_name = None
    for tools in ([a], [search, a], [b, a, search], [a, search]):
        manager = ToolManager(tools)
        name = next(name for name, entry in manager.atomic_tool_catalog().items() if entry['origin'] == 'a')
        retrieval = manager.enable_tool_retrieval(
            required=[], groups=set(), estimate_tokens=lambda schemas: len(schemas),
            threshold_tokens=100, state_store=state_store)
        if original_name is None:
            original_name = name
            retrieval.load([name], [])
        assert name == original_name
        visible = {d['function']['name'] for d in manager.tools_description}
        assert visible == {'search_tools', 'load_tools', original_name}
        assert 'a:search' in manager.tools_info[original_name]({})


def test_same_server_normalization_preserves_distinct_wire_routing():
    manager = ToolManager([make_tool('a', 'foo.bar'), make_tool('a', 'foo-bar')])
    catalog = manager.atomic_tool_catalog()
    assert len(catalog) == 2
    assert all(name.startswith('foo_bar_mcp_') for name in catalog)
    results = [manager.tools_info[name]({}) for name in catalog]
    assert any('a:foo.bar' in value for value in results)
    assert any('a:foo-bar' in value for value in results)


def test_alias_ignores_host_display_mutations_and_obeys_model_name_limit():
    wire_name = '工具.' + 'long-tool-' * 12
    tool = make_tool('a', wire_name)
    expected = next(iter(ToolManager([tool]).atomic_tool_catalog()))
    tool.__name__ = 'renamed_by_host'
    manager = ToolManager([make_tool('b'), tool])
    name = next(name for name, entry in manager.atomic_tool_catalog().items() if entry['origin'] == 'a')
    assert name == expected
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name)
    assert f'a:{wire_name}' in manager.tools_info[name]({})


def test_exact_final_alias_collision_is_rejected():
    tool = make_tool('a')
    alias = next(iter(ToolManager([tool]).atomic_tool_catalog()))

    def local() -> str:
        '''Search local documents.'''
        return 'local'

    local.__name__ = alias
    with pytest.raises(ValueError, match='Duplicate tool name'):
        ToolManager([local, tool])
