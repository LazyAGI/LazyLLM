import pytest
import unittest

from lazyllm.tools import MCPClient


class TestMCP(unittest.TestCase):

    def setUp(self):
        # for Windows system
        # use '{ "command": "cmd", "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", "./"]}'
        self.config = {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "./"
            ]
        }
        self.client = MCPClient(command_or_url=self.config["command"], args=self.config["args"])

    def test_get_tools_sync(self):
        tools = self.client.get_tools()
        assert len(tools) != 0, f"Expected at least one tool, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_get_tools_async(self):
        tools = await self.client.aget_tools()
        assert len(tools) != 0, f"Expected at least one tool, got {len(tools)}"

    @pytest.mark.asyncio
    async def test_tool_call(self):
        res = await self.client.call_tool(tool_name="list_allowed_directories", arguments={})
        assert "Tool call result:" in res
