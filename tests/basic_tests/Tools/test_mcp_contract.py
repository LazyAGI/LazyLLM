import asyncio
import json
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent
import pytest

from lazyllm.tools.agent.toolsManager import ToolManager
from lazyllm.tools.mcp.client import MCPClient
from lazyllm.tools.mcp.tool_adaptor import generate_lazyllm_tool


def wire(name='knowledge.create', schema=None):
    return SimpleNamespace(name=name, description='Manage knowledge.',
                           inputSchema=schema or {'type': 'object', 'properties': {}})


def test_empty_allowlist_does_not_discover_or_register():
    client = MCPClient('https://example.test/mcp')

    async def discover():
        pytest.fail('An empty allowlist must not contact the server')

    client.list_tools = discover
    assert client.get_tools(allowed_tools=[]) == []


@pytest.mark.parametrize('allowed, expected', [(None, ['knowledge.create', 'knowledge.search']),
                                               (['knowledge.create'], ['knowledge.create'])])
def test_allowlist_filters_wire_names_before_registration(monkeypatch, allowed, expected):
    import lazyllm.tools.mcp.client as module
    registered = []
    client = MCPClient('https://example.test/mcp')

    async def discover():
        return SimpleNamespace(tools=[wire(), wire('knowledge.search')])

    def register(client, tool):
        registered.append(tool.name)
        return tool.name

    client.list_tools = discover
    monkeypatch.setattr(module, 'generate_lazyllm_tool', register)
    assert client.get_tools(allowed_tools=allowed) == expected
    assert registered == expected


def test_structured_error_reaches_model_and_keeps_wire_identity():
    client = MCPClient('https://example.test/mcp', server_id='builtin', result_mode='structured')

    async def call_tool(name, arguments):
        return CallToolResult(content=[TextContent(type='text', text='Scope too large')], isError=True,
                              structuredContent={'code': 'QUERY_TOO_LARGE', 'allowed': 20, 'actual': 21})

    client.call_tool = call_tool
    tool = generate_lazyllm_tool(client, wire())
    assert tool.__mcp_tool_name__ == 'knowledge.create'
    manager = ToolManager([tool])
    name, entry = next(iter(manager.atomic_tool_catalog().items()))
    assert entry['origin'] == 'builtin'
    result = json.loads(manager.tools_info[name]({}))
    assert result['isError'] is True
    assert result['structuredContent'] == {'code': 'QUERY_TOO_LARGE', 'allowed': 20, 'actual': 21}
    assert result['content'][0]['text'] == 'Scope too large'


def test_legacy_result_text_is_unchanged():
    client = MCPClient('https://example.test/mcp')

    async def call_tool(name, arguments):
        return CallToolResult(content=[TextContent(type='text', text='hello')],
                              structuredContent={'value': 1}, isError=True)

    client.call_tool = call_tool
    assert generate_lazyllm_tool(client, wire())() == 'Tool call result:\nReceived text message:\nhello'


def test_structured_content_without_text_is_not_discarded():
    client = MCPClient('https://example.test/mcp', result_mode='structured')

    async def call_tool(name, arguments):
        return CallToolResult(content=[], structuredContent={'id': 'created'}, isError=False)

    client.call_tool = call_tool
    result = json.loads(generate_lazyllm_tool(client, wire())())
    assert result == {'content': [], 'structuredContent': {'id': 'created'}, 'isError': False}


def test_trusted_metadata_and_progress_do_not_enter_tool_arguments():
    from contextlib import asynccontextmanager
    calls = []
    progress = []

    async def metadata(name, arguments):
        return {'operation_key': name + ':stable'}

    async def on_progress(current, total, message):
        progress.append((current, total, message))

    client = MCPClient('https://example.test/mcp', request_metadata_provider=metadata,
                       progress_callback=on_progress)

    @asynccontextmanager
    async def session():
        class Session:
            async def call_tool(self, name, arguments, *, meta, progress_callback):
                calls.append((name, arguments, meta))
                await progress_callback(1, 2, 'Copying')
                return 'done'
        yield Session()

    client._run_session = session
    assert asyncio.run(client.call_tool('knowledge.create', {'name': 'Papers'})) == 'done'
    assert calls == [('knowledge.create', {'name': 'Papers'}, {'operation_key': 'knowledge.create:stable'})]
    assert progress == [(1, 2, 'Copying')]


def test_nested_union_schema_and_explicit_null_survive_model_adaptation():
    schema = {'type': 'object', 'properties': {
        'source': {'oneOf': [
            {'type': 'object', 'properties': {'kind': {'const': 'file'}, 'file_ref': {'type': 'string'}},
             'required': ['kind', 'file_ref']},
            {'type': 'object', 'properties': {'kind': {'const': 'url'}, 'url': {'type': 'string'}},
             'required': ['kind', 'url']}]},
        'parent_id': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}}, 'required': ['source']}
    client = MCPClient('https://example.test/mcp')
    received = []

    async def call_tool(name, arguments):
        received.append(arguments)
        return CallToolResult(content=[])

    client.call_tool = call_tool
    manager = ToolManager([generate_lazyllm_tool(client, wire(schema=schema))])
    exposed = manager.tools_description[0]['function']['parameters']
    assert exposed['properties']['source']['oneOf'] == schema['properties']['source']['oneOf']
    assert exposed['properties']['parent_id']['anyOf'] == [{'type': 'string'}, {'type': 'null'}]
    name = next(iter(manager.atomic_tool_catalog()))
    manager.tools_info[name]({'source': {'kind': 'file', 'file_ref': 'f'}, 'parent_id': None})
    assert received == [{'source': {'kind': 'file', 'file_ref': 'f'}, 'parent_id': None}]


def test_top_level_constraints_and_defs_survive_catalog():
    schema = {'type': 'object', '$defs': {'ref': {'type': 'string', 'minLength': 1}},
              'properties': {'file_ref': {'$ref': '#/$defs/ref'}, 'url': {'type': 'string'}},
              'oneOf': [{'required': ['file_ref']}, {'required': ['url']}],
              'additionalProperties': False}
    tool = generate_lazyllm_tool(MCPClient('https://example.test/mcp'), wire(schema=schema))
    exposed = ToolManager([tool]).tools_description[0]['function']['parameters']
    assert exposed == schema


def test_structured_transport_failure_is_error_and_does_not_expose_credentials():
    client = MCPClient('https://example.test/mcp', result_mode='structured')

    async def call_tool(name, arguments):
        raise TimeoutError('Authorization: Bearer private-token')

    client.call_tool = call_tool
    result = json.loads(generate_lazyllm_tool(client, wire())())
    assert result['isError'] is True
    assert result['structuredContent']['code'] == 'MCP_CALL_FAILED'
    assert result['structuredContent']['error_type'] == 'TimeoutError'
    assert 'private-token' not in json.dumps(result)


def test_sync_bridge_keeps_trusted_context_when_event_loop_is_running():
    from contextvars import ContextVar
    from lazyllm.tools.mcp.utils import patch_sync
    request = ContextVar('test_trusted_request', default=None)

    async def read():
        return request.get()

    async def invoke():
        token = request.set('call-a')
        try:
            return patch_sync(read)()
        finally:
            request.reset(token)

    assert asyncio.run(invoke()) == 'call-a'
    assert request.get() is None


def test_nullable_type_array_loads_and_preserves_original_schema():
    schema = {'type': 'object', 'properties': {
        'folder_id': {'type': ['string', 'null'], 'description': 'Optional parent'},
        'names': {'type': ['array', 'null'], 'items': {'type': 'string'}},
    }}
    tool = generate_lazyllm_tool(MCPClient('https://example.test/mcp'), wire(schema=schema))
    manager = ToolManager([tool])
    assert tool.__mcp_input_schema__ == schema
    assert manager.tools_description[0]['function']['parameters'] == schema
