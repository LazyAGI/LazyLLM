import hashlib
from types import SimpleNamespace

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
    manager = ToolManager(tools)
    catalog = manager.atomic_tool_catalog()
    expected = {f'search_mcp_{hashlib.sha256((origin + chr(0) + "search").encode()).hexdigest()[:12]}'
                for origin in ('a', 'b')}
    assert set(catalog) == expected
    assert len({entry['identity'] for entry in catalog.values()}) == 2
    assert {entry['origin'] for entry in catalog.values()} == {'a', 'b'}
    for name, entry in catalog.items():
        assert f"{entry['origin']}:search" in manager.tools_info[name]({})
    assert [tool.__name__ for tool in tools] == ['search', 'search']
    assert set(ToolManager(list(reversed(tools))).atomic_tool_catalog()) == expected
    assert list(ToolManager([tools[0]]).atomic_tool_catalog()) == ['search']
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
