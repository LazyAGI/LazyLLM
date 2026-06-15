# flake8: noqa E501
import importlib
import functools
from .. import utils
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))

add_english_doc('MCPClient', '''\
MCP client that can be used to connect to an MCP server. It supports both local servers (through stdio client) and remote servers (through sse client or streamable-http client).

If the 'command_or_url' is a url string (started with 'http' or 'https'), a remote server will be connected, otherwise a local server will be started and connected.

Args:
    command_or_url (str): The command or url string, which will be used to start a local server or connect to a remote server.
    args (list[str], optional): Arguments list used for starting a local server, if you want to connect to a remote server, this argument is not needed. (default is [])
    env (dict[str, str], optional): Environment variables dictionary used in tools, for example some api keys. (default is None)
    headers(dict[str, Any], optional): HTTP headers used in http client connection. (default is None)
    timeout (float, optional): Timeout for http client connection, in seconds. (default is 5)
    transport (Literal, optional): Transport protocol to use. One of 'auto', 'stdio', 'sse', 'streamable-http'. (default is 'auto')

        - 'auto': For http(s) URLs, selects 'streamable-http'. For non-http commands, selects 'stdio'. SSE is NOT auto-detected — use explicit transport='sse' for legacy SSE servers.
        - 'stdio': Launch and communicate with a local MCP server via standard input/output.
        - 'sse': Connect to a remote MCP server using the legacy SSE protocol (deprecated by MCP spec, but still supported for backward compatibility).
        - 'streamable-http': Connect to a remote MCP server using the new Streamable HTTP protocol.
''')

add_chinese_doc('MCPClient', '''\
MCP客户端，用于连接MCP服务器。同时支持本地服务器（通过 stdio）和远程服务器（通过 SSE 或 Streamable HTTP）。

如果传入的 'command_or_url' 是一个 URL 字符串（以 'http' 或 'https' 开头），则将连接到远程服务器；否则，将启动并连接到本地服务器。


Args:
    command_or_url (str): 用于启动本地服务器或连接远程服务器的命令或 URL 字符串。
    args (list[str], optional): 用于启动本地服务器的参数列表；如果要连接远程服务器，则无需此参数。（默认值为[]）
    env (dict[str, str], optional): 工具中使用的环境变量，例如一些 API 密钥。（默认值为None）
    headers(dict[str, Any], optional): 用于 HTTP 客户端连接的请求头。（默认值为None）
    timeout (float, optional): HTTP 客户端连接的超时时间，单位为秒。(默认值为5)
    transport (Literal, optional): 传输协议。可选值为 'auto'、'stdio'、'sse'、'streamable-http'。(默认值为 'auto')

        - 'auto'：对于 http(s) URL，选择 'streamable-http'。对于非 http 命令，选择 'stdio'。SSE 不会被自动检测——老版本 SSE 服务器请显式设置 transport='sse'。
        - 'stdio'：通过标准输入输出启动和通信本地 MCP 服务器。
        - 'sse'：使用旧版 SSE 协议连接远程 MCP 服务器（已被 MCP 规范废弃，仍保留以兼容老服务）。
        - 'streamable-http'：使用新版 Streamable HTTP 协议连接远程 MCP 服务器。
''')


add_english_doc('MCPClient.call_tool', '''\
Calls one of the tools provided in the toolset of the connected MCP server via the MCP client and returns the result.

Args:
    tool_name (str): The name of the tool.
    arguments (dict): The parameters for the tool.
''')

add_chinese_doc('MCPClient.call_tool', '''\
通过MCP客户端调用连接的MCP服务器提供的工具集中的某一个工具，并返回结果。

Args:
    tool_name (str): 工具名称。
    arguments (dict): 工具传参。
''')


add_english_doc('MCPClient.list_tools', '''\
Retrieve the list of tools from the currently connected MCP client.

**Returns:**\n
- Any: The list of tools returned by the MCP client.
''')

add_chinese_doc('MCPClient.list_tools', '''\
获取当前连接的 MCP 客户端的工具列表。

**Returns:**\n
- Any: MCP 客户端返回的工具列表。
''')


add_english_doc('MCPClient.get_tools', '''\
Retrieve a filtered list of tools from the MCP client.

Args:
    allowed_tools (Optional[list[str]]): List of tool names to filter. If None, all tools are returned.

**Returns:**\n
- Any: List of tools that match the filter criteria.
''')

add_chinese_doc('MCPClient.get_tools', '''\
从 MCP 客户端获取经过筛选的工具列表。

Args:
    allowed_tools (Optional[list[str]]): 要筛选的工具名称列表，若为 None，则返回所有工具。

**Returns:**\n
- Any: 符合筛选条件的工具列表。
''')


add_english_doc('MCPClient.deploy', '''\
Deploys the MCP client with the specified SSE server settings asynchronously.

Args:
    sse_settings (SseServerSettings): Configuration settings for the SSE server.
''')

add_chinese_doc('MCPClient.deploy', '''\
使用指定的 SSE 服务器设置异步部署 MCP 客户端。

Args:
    sse_settings (SseServerSettings): SSE 服务器的配置设置。
''')


add_english_doc('MCPClient.aget_tools', '''\
Used to convert the tool set from the MCP server into a list of functions available for LazyLLM and return them.

The allowed_tools parameter is used to specify the list of tools to be returned. If None, all tools will be returned.

Args:
    allowed_tools (list[str], optional): The list of tools expected to be returned. Defaults to None, meaning that all tools will be returned.
''')

add_chinese_doc('MCPClient.aget_tools', '''\
用于将MCP服务器中的工具集转换为LazyLLM可用的函数列表，并返回。

allowed_tools参数用于指定要返回的工具列表，默认为None，表示返回所有工具。

Args:
    allowed_tools (list[str], optional): 期望返回的工具列表，默认为None，表示返回所有工具。
''')


add_example('MCPClient', '''\
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
>>> print(agent("Write a Chinese poem about the moon, and save it to a file named 'moon.txt'."))
''')


# ---------------------------------------------------------------------------- #

# mcp/tool_adaptor.py

add_english_doc('mcp.tool_adaptor.generate_lazyllm_tool', '''\
Dynamically build a function for the LazyLLM agent based on a tool provided by the MCP server.

Args:
    client (mcp.ClientSession): MCP client which connects to the MCP server.
    mcp_tool (mcp.types.Tool): A tool provided by the MCP server.
''')

add_chinese_doc('mcp.tool_adaptor.generate_lazyllm_tool', '''\
将 MCP 服务器提供的工具转换为 LazyLLM 代理使用的函数。

Args:
    client (mcp.ClientSession): 连接到MCP服务器的MCP客户端。
    mcp_tool (mcp.types.Tool): 由MCP服务器提供的工具。
''')


