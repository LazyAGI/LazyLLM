import importlib.util

from typing import Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from lazyllm.thirdparty import mcp

from .utils import patch_sync
from .tool_adaptor import generate_lazyllm_tool
from .deploy import SseServerSettings, start_sse_server


class MCPClient(object):
    """MCP client that can be used to connect to an MCP server. It supports both local servers (through stdio client) and remote servers (through sse client).

If the 'command_or_url' is a url string (started with 'http' or 'https'), a remote server will be connected, otherwise a local server will be started and connected.

Args:
    command_or_url (str): The command or url string, which will be used to start a local server or connect to a remote server.
    args (list[str], optional): Arguments list used for starting a local server, if you want to connect to a remote server, this argument is not needed. (default is [])
    env (dict[str, str], optional): Environment variables dictionary used in tools, for example some api keys. (default is None)
    headers(dict[str, Any], optional): HTTP headers used in sse client connection. (default is None)
    timeout (float, optional): Timeout for sse client connection, in seconds. (default is 5)


Examples:
    >>> from lazyllm.tools import MCPClient
    >>> mcp_server_configs = {
    ...     "filesystem": {
    ...         "command": "npx",
    ...         "args": [
    ...             "-y",
    ...             "@modelcontextprotocol/server-filesystem",
    ...             "./",
    ...         ]
    ...     }
    ... }
    >>> file_sys_config = mcp_server_configs["filesystem"]
    >>> file_client = MCPClient(
    ...     command_or_url=file_sys_config["command"],
    ...     args=file_sys_config["args"],
    ... )
    >>> from lazyllm import OnlineChatModule
    >>> from lazyllm.tools.agent.reactAgent import ReactAgent
    >>> llm=OnlineChatModule(source="deepseek", stream=False)
    >>> agent = ReactAgent(llm.share(), file_client.get_tools())
    >>> print(agent("Write a Chinese poem about the moon, and save it to a file named 'moon.txt".))
    """
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
            spec = importlib.util.find_spec("mcp.client.sse")
            if spec is None:
                raise ImportError(
                    "Please install mcp to use mcp module. "
                    "You can install it with `pip install mcp`"
                )
            sse_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sse_module)
            sse_client = sse_module.sse_client

            async with sse_client(
                url=self._command_or_url,
                headers=self._headers,
                timeout=self._timeout
            ) as streams:
                async with mcp.ClientSession(*streams) as session:
                    await session.initialize()
                    yield session
        else:
            server_parameters = mcp.StdioServerParameters(
                command=self._command_or_url, args=self._args, env=self._env
            )
            async with mcp.stdio_client(server_parameters) as streams:
                async with mcp.ClientSession(*streams) as session:
                    await session.initialize()
                    yield session

    async def call_tool(self, tool_name: str, arguments: dict):
        """Calls one of the tools provided in the toolset of the connected MCP server via the MCP client and returns the result.

Args:
    tool_name (str): The name of the tool.
    arguments (dict): The parameters for the tool.
"""
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        """Retrieves the list of tools from the currently connected MCP client.
"""
        async with self._run_session() as session:
            return await session.list_tools()

    async def aget_tools(self, allowed_tools: list[str] = None):
        """Used to convert the tool set from the MCP server into a list of functions available for LazyLLM and return them.

The allowed_tools parameter is used to specify the list of tools to be returned. If None, all tools will be returned.

Args: 
    allowed_tools (list[str], optional): The list of tools expected to be returned. Defaults to None, meaning that all tools will be returned.
"""
        res = await self.list_tools()
        mcp_tools = getattr(res, "tools", [])
        if allowed_tools:
            mcp_tools = [tool for tool in mcp_tools if tool.name in allowed_tools]

        return [generate_lazyllm_tool(self, tool) for tool in mcp_tools]

    def get_tools(self, allowed_tools: list[str] = None):
        return patch_sync(self.aget_tools)(allowed_tools=allowed_tools)

    async def deploy(self, sse_settings: SseServerSettings):
        async with self._run_session() as session:
            await start_sse_server(session, sse_settings)
