import anyio
import pytest

pytest.importorskip('mcp')

from mcp import ClientSession, types  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

from lazyllm.tools.mcp.deploy import _create_proxy_server  # noqa: E402


class FakeRemoteSession(object):
    '''Stands in for the ClientSession of the MCP server that `lazyllm deploy mcp_server` proxies.'''

    def __init__(self):
        self.calls = []

    async def initialize(self):
        return types.InitializeResult(
            protocolVersion=types.LATEST_PROTOCOL_VERSION,
            capabilities=types.ServerCapabilities(
                prompts=types.PromptsCapability(), resources=types.ResourcesCapability(),
                logging=types.LoggingCapability(), tools=types.ToolsCapability(),
            ),
            serverInfo=types.Implementation(name='remote', version='1.0'),
        )

    async def list_prompts(self):
        return types.ListPromptsResult(prompts=[types.Prompt(name='greet')])

    async def get_prompt(self, name, arguments):
        text = f'{name}: {arguments["who"]}'
        return types.GetPromptResult(
            messages=[types.PromptMessage(role='user', content=types.TextContent(type='text', text=text))])

    async def subscribe_resource(self, uri):
        self.calls.append(('subscribe', str(uri)))
        return types.EmptyResult()

    async def unsubscribe_resource(self, uri):
        self.calls.append(('unsubscribe', str(uri)))
        return types.EmptyResult()

    async def list_resources(self):
        return types.ListResourcesResult(resources=[types.Resource(uri='demo://hello', name='hello')])

    async def read_resource(self, uri):
        return types.ReadResourceResult(contents=[types.TextResourceContents(uri=str(uri), text='hello')])

    async def set_logging_level(self, level):
        self.calls.append(('set_level', level))
        return types.EmptyResult()

    async def list_tools(self):
        return types.ListToolsResult(tools=[types.Tool(name='add', inputSchema={'type': 'object'})])

    async def call_tool(self, name, arguments):
        if name != 'add':
            raise ValueError(f'unknown tool {name}')
        return types.CallToolResult(content=[types.TextContent(type='text', text=str(arguments['a'] + arguments['b']))])

    async def complete(self, ref, argument):
        return types.CompleteResult(completion=types.Completion(values=[argument['value'] + '1']))

    async def send_progress_notification(self, progress_token, progress, total):
        self.calls.append(('progress', progress_token, progress, total))


def _field(model, name):
    # mcp 2.x only exposes snake_case attributes; the wire (alias) names work on both majors.
    return model.model_dump(by_alias=True)[name]


def test_proxy_server_forwards_requests():
    remote = FakeRemoteSession()

    async def run():
        server = await _create_proxy_server(remote)
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                tg.start_soon(lambda: server.run(*server_streams, server.create_initialization_options()))
                async with ClientSession(*client_streams) as session:
                    init = await session.initialize()
                    assert _field(init, 'serverInfo')['name'] == 'remote'

                    assert [t.name for t in (await session.list_tools()).tools] == ['add']
                    result = await session.call_tool('add', {'a': 2, 'b': 3})
                    assert result.content[0].text == '5' and not _field(result, 'isError')
                    result = await session.call_tool('missing', {})
                    assert _field(result, 'isError') and 'unknown tool missing' in result.content[0].text

                    assert [p.name for p in (await session.list_prompts()).prompts] == ['greet']
                    prompt = await session.get_prompt('greet', {'who': 'lazyllm'})
                    assert prompt.messages[0].content.text == 'greet: lazyllm'

                    assert [str(r.uri) for r in (await session.list_resources()).resources] == ['demo://hello']
                    assert (await session.read_resource('demo://hello')).contents[0].text == 'hello'
                    await session.subscribe_resource('demo://hello')
                    await session.unsubscribe_resource('demo://hello')
                    await session.set_logging_level('info')

                    completion = await session.complete(types.PromptReference(type='ref/prompt', name='greet'),
                                                        {'name': 'who', 'value': 'lazy'})
                    assert completion.completion.values == ['lazy1']

                    await session.send_progress_notification('token', 0.5, 1.0)
                    with anyio.fail_after(5):
                        while not any(c[0] == 'progress' for c in remote.calls):
                            await anyio.sleep(0.01)
                tg.cancel_scope.cancel()

    anyio.run(run)
    assert ('subscribe', 'demo://hello') in remote.calls
    assert ('unsubscribe', 'demo://hello') in remote.calls
    assert ('set_level', 'info') in remote.calls
    assert ('progress', 'token', 0.5, 1.0) in remote.calls
