import copy

import pytest
from types import SimpleNamespace

from lazyllm.tools import get_tool_runtime_metadata
from lazyllm.tools.agent.tool_runtime import _set_tool_runtime_metadata
from lazyllm.tools.agent.toolsManager import ToolManager, fc_register
from lazyllm.tools.mcp import MCPClient
from lazyllm.tools.mcp.tool_adaptor import generate_lazyllm_tool
from mcp.types import CallToolResult, TextContent


def make_tool(server_id='', name='search', *, server_name=None, tool_name_prefix=''):
    client = MCPClient('https://example.test/mcp', server_id=server_id,
                       server_name=server_id if server_name is None else server_name,
                       tool_name_prefix=tool_name_prefix)

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
    assert expected == {'mcp_a_search', 'mcp_b_search'}
    assert len({entry['identity'] for entry in catalog.values()}) == 2
    assert {entry['origin'] for entry in catalog.values()} == {'a', 'b'}
    for name, entry in catalog.items():
        assert f"{entry['origin']}:search" in manager.tools_info[name]({})
    assert [tool.__name__ for tool in tools] == ['mcp_a_search', 'mcp_b_search']
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


def test_no_id_adapter_allows_host_identity_enrichment():
    tool = make_tool(name='search.documents')
    identity = 'mcp:v1:host-security-digest'
    _set_tool_runtime_metadata(tool, {
        'tool_source': 'mcp', 'tool_origin': 'server-a', 'tool_identity': identity,
    })
    manager = ToolManager([tool])
    name, entry = next(iter(manager.atomic_tool_catalog().items()))
    assert name == 'search_documents'
    assert entry['origin'] == 'server-a'
    assert entry['identity'] == identity
    assert ':search.documents' in manager.tools_info[name]({})
    assert manager.prepare_tool_calls({'function': {'name': name, 'arguments': {}}})[0].tool_identity == identity


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
    assert next(name for name, entry in catalog.items() if entry['source'] == 'mcp') == (
        'mcp_a_mcp_registration_search')


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


def test_same_server_normalization_collision_is_rejected():
    with pytest.raises(ValueError, match='Duplicate tool name.*mcp_a_foo_bar'):
        ToolManager([make_tool('a', 'foo.bar'), make_tool('a', 'foo-bar')])


def test_same_normalized_server_name_collision_is_rejected():
    with pytest.raises(ValueError, match='Duplicate tool name.*mcp_team_a_search'):
        ToolManager([make_tool('a', server_name='Team-A'), make_tool('b', server_name='Team A')])


@pytest.mark.parametrize(('server', 'wire', 'prefix', 'expected'), [
    ('kbmanager', 'knowledge.list', 'knowledge.', 'mcp_kbmanager_list'),
    ('kbmanager', 'knowledge.document.list', 'knowledge.', 'mcp_kbmanager_document_list'),
    ('browser', 'browser.open', 'browser.', 'mcp_browser_open'),
    ('Notion', 'search', '', 'mcp_notion_search'),
    (' __Team-A / Docs__ ', 'search.Documents', '', 'mcp_team_a_docs_search_Documents'),
    ('kbmanager', 'other.list', 'knowledge.', 'mcp_kbmanager_other_list'),
])
def test_readable_names_preserve_original_wire_routing(server, wire, prefix, expected):
    tool = make_tool('a', wire, server_name=server, tool_name_prefix=prefix)
    manager = ToolManager([tool])
    assert set(manager.atomic_tool_catalog()) == {expected}
    assert tool.__mcp_tool_name__ == wire
    assert f'a:{wire}' in manager.tools_info[expected]({})
    assert tool.__name__ == expected


@pytest.mark.parametrize(('server', 'wire', 'prefix'), [
    ('---', 'search', ''), ('知识库', 'search', ''), ('a', '', ''),
    ('a', '---', ''), ('a', 'knowledge.', 'knowledge.'),
    ('a' * 60, 'list', ''), ('a', 'long-tool-' * 12, ''),
])
def test_invalid_or_long_names_are_rejected(server, wire, prefix):
    with pytest.raises(ValueError, match='MCP tool name'):
        make_tool('a', wire, server_name=server, tool_name_prefix=prefix)


def test_name_at_model_limit_is_not_truncated():
    name = 'x' * 58
    tool = make_tool('a', name)
    assert tool.__name__ == 'mcp_a_' + name
    assert len(tool.__name__) == 64


def test_exact_final_alias_collision_is_rejected():
    tool = make_tool('a')
    alias = next(iter(ToolManager([tool]).atomic_tool_catalog()))

    def local() -> str:
        '''Search local documents.'''
        return 'local'

    local.__name__ = alias
    with pytest.raises(ValueError, match='Duplicate tool name'):
        ToolManager([local, tool])


def test_legacy_identified_client_does_not_gain_a_hash_name():
    tool = make_tool('a', 'search.documents', server_name='')
    identity = get_tool_runtime_metadata(tool).tool_identity
    manager = ToolManager([tool])
    assert set(manager.atomic_tool_catalog()) == {'search_documents'}
    assert manager.atomic_tool_catalog()['search_documents']['identity'] == identity
    assert 'a:search.documents' in manager.tools_info['search_documents']({})


def test_manual_mcp_metadata_does_not_rename_callable():
    @fc_register(tool_source='mcp', tool_origin='server-a', tool_identity='server-a/search')
    def search() -> str:
        '''Search documents.'''
        return 'remote result'

    manager = ToolManager([search])
    assert set(manager.atomic_tool_catalog()) == {'search'}
    assert manager.tools_info['search']({}) == 'remote result'


def _retrieval(tools, store):
    manager = ToolManager(tools)
    controller = manager.enable_tool_retrieval(required=[], groups=set(), estimate_tokens=len,
                                               threshold_tokens=100, state_store=store)
    return manager, controller


def test_loaded_mcp_records_its_identity_and_drops_replaced_server():
    store = RetrievalStateStore()
    manager, controller = _retrieval([make_tool('a', server_name='Docs')], store)
    controller.load(['mcp_docs_search'], [])
    assert store.state['mcp_bindings'] == {'mcp_docs_search': '["a", "search"]'}
    replacement, controller = _retrieval([make_tool('b', server_name='Docs')], store)
    assert {item['function']['name'] for item in replacement.tools_description} == {'search_tools', 'load_tools'}
    controller.initialize()
    assert store.state['mcp_bindings'] == {}
    controller.load(['mcp_docs_search'], [])
    assert store.state['mcp_bindings'] == {'mcp_docs_search': '["b", "search"]'}


def test_endpoint_identity_change_drops_optional_load():
    def tool(identity):
        result = make_tool(server_name='Docs')
        _set_tool_runtime_metadata(result, {'tool_origin': 'same-server', 'tool_identity': identity})
        return result

    store = RetrievalStateStore()
    _, controller = _retrieval([tool('endpoint-old')], store)
    controller.load(['mcp_docs_search'], [])
    manager, _ = _retrieval([tool('endpoint-new')], store)
    assert {item['function']['name'] for item in manager.tools_description} == {'search_tools', 'load_tools'}


def test_reusing_mcp_name_for_local_tool_does_not_restore_its_load():
    store = RetrievalStateStore()
    _, controller = _retrieval([make_tool('a', server_name='Docs')], store)
    controller.load(['mcp_docs_search'], [])

    def mcp_docs_search() -> str:
        '''Search local documents.'''
        return 'local'

    manager, _ = _retrieval([mcp_docs_search], store)
    assert {item['function']['name'] for item in manager.tools_description} == {'search_tools', 'load_tools'}


def test_unloading_mcp_removes_binding_and_missing_binding_never_restores():
    store = RetrievalStateStore()
    manager, controller = _retrieval([make_tool('a', server_name='Docs')], store)
    controller.load(['mcp_docs_search'], [])
    controller.load([], ['mcp_docs_search'])
    assert store.state['mcp_bindings'] == {}
    store.state['loaded'].append('mcp_docs_search')
    assert {item['function']['name'] for item in manager.tools_description} == {'search_tools', 'load_tools'}


def test_discovery_rejects_normalized_collisions_before_returning_tools(monkeypatch):
    import asyncio
    client = MCPClient('https://example.test/mcp', server_name='Docs')

    async def list_tools():
        tools = [SimpleNamespace(name=name, description='Search.',
                                 inputSchema={'type': 'object', 'properties': {}})
                 for name in ['foo.bar', 'foo-bar']]
        return SimpleNamespace(tools=tools)

    monkeypatch.setattr(client, 'list_tools', list_tools)
    with pytest.raises(ValueError, match='MCP tool name.*collision'):
        asyncio.run(client.aget_tools())


def test_old_hash_names_are_not_callable_or_guessed_in_skill_dependencies():
    manager, controller = _retrieval([make_tool('kbmanager', 'knowledge.list', tool_name_prefix='knowledge.')],
                                     RetrievalStateStore())
    old_name = 'knowledge_list_mcp_1219d1082aac'
    result = manager([{'id': 'old', 'function': {'name': old_name, 'arguments': '{}'}}])
    assert result[0]['ok'] is False
    outcome = controller.load_skill('custom-skill', [old_name])
    assert outcome['loaded'] == []
    assert outcome['unavailable']
    assert old_name in str(outcome['unavailable'])
    assert {item['function']['name'] for item in manager.tools_description} == {'search_tools', 'load_tools'}
