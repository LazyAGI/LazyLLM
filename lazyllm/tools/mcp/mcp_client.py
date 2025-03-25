from urllib.parse import urlparse
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters


class MCPClient(ClientSession):
    def __init__(
        self, command_or_url: str, args: list[str] = [], env: dict[str, str] = None
    ):
        self._command_or_url = command_or_url
        self._args = args
        self._env = env

    @asynccontextmanager
    async def _run_session(self):
        if urlparse(self._command_or_url).scheme in ("http", "https"):
            async with sse_client(self._command_or_url) as streams:
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