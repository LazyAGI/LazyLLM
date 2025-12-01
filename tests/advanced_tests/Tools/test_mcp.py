import pytest
import unittest
import platform

from lazyllm.tools import MCPClient


class TestMCP(unittest.TestCase):
    def setUp(self):
        if platform.system() == "Windows":
            self.config = {
                "command": "cmd",
                "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", "./"]
            }
        else:
            self.config = {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"]
            }
        self.client = MCPClient(command_or_url=self.config["command"], args=self.config["args"])

    def test_get_tools_sync(self):
        tools = self.client.get_tools(allowed_tools=["list_allowed_directories"])
        assert len(tools) == 1, f"Expected one tool 'list_allowed_directories', got {len(tools)}"

    @pytest.mark.asyncio
    async def test_get_tools_async(self):
        tools = await self.client.aget_tools(allowed_tools=["list_allowed_directories"])
        assert len(tools) == 1, f"Expected one tool 'list_allowed_directories', got {len(tools)}"

    def test_tool_call(self):
        tool = self.client.get_tools(allowed_tools=["list_allowed_directories"])[0]
        res = tool()
        assert "Tool call result:" in res
