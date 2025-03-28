from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from typing import Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from .tool_adaptor import generate_lazyllm_tool


class MCPClient(ClientSession):
    def __init__(
        self,
        command_or_url: str,
        args: list[str] = [],
        env: dict[str, str] = None,
        headers: dict[str, Any] = None,
        timeout: float = 5,
    ):
        self._command_or_url = command_or_url
        self._args = args
        self._env = env
        self._headers = headers
        self._timeout = timeout
    
    @asynccontextmanager
    async def _run_session(self):
        if urlparse(self._command_or_url).scheme in ("http", "https"):
            async with sse_client(url=self._command_or_url, headers=self._headers, timeout=self._timeout) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session
        else:
            server_parameters = StdioServerParameters(
                command=self._command_or_url, args=self._args, env=self._env
            )
            async with stdio_client(server_parameters) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    yield session

    async def call_tool(self, tool_name: str, arguments: dict):
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        async with self._run_session() as session:
            return await session.list_tools()
    
    async def get_tools(self, allowed_tools: list[str] = None):
        res = await self.list_tools()
        mcp_tools = getattr(res, "tools", [])
        if allowed_tools:
            mcp_tools = [tool for tool in mcp_tools if tool.name in allowed_tools]
        
        return [generate_lazyllm_tool(self, tool) for tool in mcp_tools]
    
