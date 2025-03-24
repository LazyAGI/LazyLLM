from lazyllm import TrainableModule
from lazyllm.tools.agent.reactAgent import ReactAgent
from lazyllm.tools.mcp import MCPClient, McpToolAdaptor


test_configs = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "./",
        ]
    }
}


def main():
    test_config = test_configs["mcpServers"]["fetch"]
    test_client = MCPClient(
        command_or_url=test_config["command"],
        args=test_config["args"],
        env=test_config.get("env")
    )
    mcp_spec = McpToolAdaptor(test_client)
    tools = mcp_spec.tool_list()
    agent = ReactAgent(
        llm=TrainableModule('internlm2-chat-7b'),
        tools=tools,
    )
    print(agent("我的目录下有哪些文件？"))


if __name__ == "__main__":
    main()


