from typing import Any, Optional, Literal
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from lazyllm.thirdparty import httpx, mcp

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
    ):
        self._command_or_url = command_or_url
        self._args = args or []
        self._env = env
        self._headers = headers
        self._timeout = timeout
        self._transport = transport
        self._probed_transport: Optional[str] = None

    async def _resolve_transport(self) -> str:
        if self._transport != 'auto':
            return self._transport
        if self._probed_transport is not None:
            return self._probed_transport
        self._probed_transport = await self._probe_transport()
        return self._probed_transport

    async def _probe_transport(self) -> str:
        url = self._command_or_url
        if urlparse(url).scheme not in ('http', 'https'):
            return 'stdio'

        timeout = httpx.Timeout(min(self._timeout, 5))
        headers = {
            **(self._headers or {}),
            'Accept': 'application/json, text/event-stream',
            'Content-Type': 'application/json',
        }
        body = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-03-26',
                'capabilities': {},
                'clientInfo': {'name': 'lazyllm', 'version': '1.0'},
            },
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=body, headers=headers)
                if r.status_code < 500:
                    return 'streamable-http'
        except Exception:
            pass

        raise RuntimeError(
            f"Cannot auto-detect MCP transport for '{url}'. "
            f"If this server uses the legacy SSE transport, set transport='sse'."
        )

    @asynccontextmanager
    async def _run_session(self):
        transport = await self._resolve_transport()

        # ───────── stdio ─────────
        if transport == 'stdio':
            server_parameters = mcp.StdioServerParameters(
                command=self._command_or_url, args=self._args, env=self._env
            )
            async with mcp.stdio_client(server_parameters) as (read_stream, write_stream):
                async with mcp.ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        # ───────── Streamable HTTP (new standard) ─────────
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

            async with create_mcp_http_client(
                headers=self._headers or None,
                timeout=httpx.Timeout(self._timeout),
            ) as http_client:
                async with streamable_http_client(
                    url=self._command_or_url,
                    http_client=http_client,
                ) as (read_stream, write_stream, _get_session_id):
                    async with mcp.ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session

        # ───────── Legacy SSE (backward compatible) ─────────
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

    async def call_tool(self, tool_name: str, arguments: dict):
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        async with self._run_session() as session:
            return await session.list_tools()

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
