import importlib.util

from typing import Any
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from lazyllm.thirdparty import mcp

from .utils import patch_sync
from .tool_adaptor import generate_lazyllm_tool
from .deploy import SseServerSettings, start_sse_server


class MCPClient(object):
    """MCP客户端，用于连接MCP服务器。同时支持本地服务器（通过stdio client）和sse服务器（通过sse client）。

如果传入的 'command_or_url' 是一个 URL 字符串（以 'http' 或 'https' 开头），则将连接到远程服务器；否则，将启动并连接到本地服务器。

Args:
    command_or_url (str): 用于启动本地服务器或连接远程服务器的命令或 URL 字符串。
    args (list[str], optional): 用于启动本地服务器的参数列表；如果要连接远程服务器，则无需此参数。（默认值为[]）
    env (dict[str, str], optional): 工具中使用的环境变量，例如一些 API 密钥。（默认值为None）
    headers(dict[str, Any], optional): 用于sse客户端连接的HTTP头。（默认值为None）
    timeout (float, optional): sse客户端连接的超时时间，单位为秒。(默认值为5)


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
        """通过MCP客户端调用连接的MCP服务器提供的工具集中的某一个工具，并返回结果。

Args:
    tool_name (str): 工具名称。
    arguments (dict): 工具传参。
"""
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        """获取当前连接MCP客户端的工具列表。
"""
        async with self._run_session() as session:
            return await session.list_tools()

    async def aget_tools(self, allowed_tools: list[str] = None):
        """用于将MCP服务器中的工具集转换为LazyLLM可用的函数列表，并返回。

allowed_tools参数用于指定要返回的工具列表，默认为None，表示返回所有工具。

Args:
    allowed_tools (list[str], optional): 期望返回的工具列表，默认为None，表示返回所有工具。
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
