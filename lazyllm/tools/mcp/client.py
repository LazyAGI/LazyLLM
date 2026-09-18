from typing import Any, Optional, Literal, Awaitable, Callable
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from lazyllm.thirdparty import mcp

from .utils import patch_sync
from .tool_adaptor import generate_lazyllm_tool
from .deploy import SseServerSettings, start_sse_server


class MCPClient(object):
    def __init__(
        self,
        command_or_url: str,
        args: Optional[list[str]] = None,
        env: dict[str, str] = None,
        headers: dict[str, Any] = None,
        timeout: float = 5,
        transport: Literal['auto', 'stdio', 'sse', 'streamable-http'] = 'auto',
        auth_provider: Optional[Callable[[], Awaitable[dict[str, str]]]] = None,
        auth_recovery: Optional[Callable[[], Awaitable[bool]]] = None,
    ):
        self._command_or_url = command_or_url
        self._args = args or []
        self._env = env
        self._headers = headers
        self._timeout = timeout
        self._transport = transport
        self._auth_provider = auth_provider
        self._auth_recovery = auth_recovery
        if (auth_provider or auth_recovery) and self._resolve_transport() != 'streamable-http':
            raise ValueError('MCP auth callbacks require streamable-http transport')

    def _resolve_transport(self) -> str:
        if self._transport != 'auto':
            return self._transport
        if urlparse(self._command_or_url).scheme in ('http', 'https'):
            return 'streamable-http'
        return 'stdio'

    @asynccontextmanager
    async def _run_session(self):
        transport = self._resolve_transport()

        if transport == 'stdio':
            server_parameters = mcp.StdioServerParameters(
                command=self._command_or_url, args=self._args, env=self._env
            )
            async with mcp.stdio_client(server_parameters) as (read_stream, write_stream):
                async with mcp.ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        elif transport == 'streamable-http':
            import importlib.util
            spec = importlib.util.find_spec('mcp.client.streamable_http')
            if spec is None:
                raise ImportError(
                    'Please install mcp to use mcp module. '
                    'You can install it with `pip install mcp`'
                )
            streamable_http_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(streamable_http_module)
            streamable_http_client = streamable_http_module.streamable_http_client
            create_mcp_http_client = streamable_http_module.create_mcp_http_client

            headers = dict(self._headers or {})
            if self._auth_provider:
                headers.update(await self._auth_provider())
            async with create_mcp_http_client(
                headers=headers or None,
                # Let the MCP SDK construct its own timeout object. Some
                # releases vendor httpx as httpx2, which cannot consume a
                # Timeout instance created by LazyLLM's httpx shim.
                timeout=self._timeout,
            ) as http_client:
                if self._auth_provider:
                    # Newer SDKs follow same-origin redirects themselves. Reject
                    # responses before the SDK can resend credentials.
                    http_client.follow_redirects = False

                    async def reject_redirect(response):
                        if 300 <= response.status_code < 400:
                            raise RuntimeError('MCP HTTP redirects are not allowed')
                    http_client.event_hooks['response'].append(reject_redirect)
                async with streamable_http_client(
                    url=self._command_or_url,
                    http_client=http_client,
                ) as streams:
                    # mcp SDK releases have returned both a 2-tuple and a
                    # 3-tuple here. Only the read/write streams are required.
                    read_stream, write_stream = streams[:2]
                    async with mcp.ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
        else:  # 'sse'
            import importlib.util
            spec = importlib.util.find_spec('mcp.client.sse')
            if spec is None:
                raise ImportError(
                    'Please install mcp to use mcp module. '
                    'You can install it with `pip install mcp`'
                )
            sse_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sse_module)
            sse_client = sse_module.sse_client

            async with sse_client(
                url=self._command_or_url,
                headers=self._headers,
                timeout=self._timeout,
            ) as (read_stream, write_stream):
                async with mcp.ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

    @staticmethod
    def _is_unauthorized(error):
        # AnyIO may wrap transport failures in exception groups. Mixed groups
        # are ambiguous and must never replay a potentially mutating tool.
        children = getattr(error, 'exceptions', None)
        if children is not None:
            return bool(children) and all(MCPClient._is_unauthorized(child) for child in children)
        return (type(error).__name__ == 'HTTPStatusError'
                and type(error).__module__.split('.')[0] in ('httpx', 'httpx2')
                and getattr(getattr(error, 'response', None), 'status_code', None) == 401)

    async def _request(self, method, *args):
        for attempt in range(2):
            completed = False
            try:
                async with self._run_session() as session:
                    result = await getattr(session, method)(*args)
                    completed = True
                    return result
            except Exception as error:
                if (completed or attempt or not self._auth_recovery or not self._is_unauthorized(error)
                        or not await self._auth_recovery()):
                    raise

    async def call_tool(self, tool_name: str, arguments: dict):
        return await self._request('call_tool', tool_name, arguments)

    async def list_tools(self):
        return await self._request('list_tools')

    async def aget_tools(self, allowed_tools: list[str] = None):
        res = await self.list_tools()
        mcp_tools = getattr(res, 'tools', [])
        if allowed_tools:
            mcp_tools = [tool for tool in mcp_tools if tool.name in allowed_tools]

        return [generate_lazyllm_tool(self, tool) for tool in mcp_tools]

    def get_tools(self, allowed_tools: list[str] = None):
        return patch_sync(self.aget_tools)(allowed_tools=allowed_tools)

    async def deploy(self, sse_settings: SseServerSettings):
        async with self._run_session() as session:
            await start_sse_server(session, sse_settings)
