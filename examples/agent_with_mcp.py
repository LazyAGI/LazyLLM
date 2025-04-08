import lazyllm
import asyncio

# Before running, set the environment variable:
#
# 1. `export LAZYLLM_DEEPSEEK_API_KEY=xxxx`: the API key of DeepSeek.
#     You can apply for the API key at https://platform.deepseek.com/
#     Also supports other API keys:
#       - LAZYLLM_OPENAI_API_KEY: the API key of OpenAI, default model "gpt-3.5-turbo", `source="openai"`.
#           You can apply for the API key at https://openai.com/index/openai-api/
#       - LAZYLLM_KIMI_API_KEY: the API key of Moonshot AI, default model "moonshot-v1-8k", `source="kimi"`.
#           You can apply for the API key at https://platform.moonshot.cn/console
#       - LAZYLLM_QWEN_API_KEY: the API key of Alibaba Cloud, default model "qwen-plus", `source="qwen"`.
#           You can apply for the API key at https://home.console.aliyun.com/
#       - LAZYLLM_SENSENOVA_API_KEY: the API key of SenseTime, default model "SenseChat-5", `source="sensenova"`.
#                                  You also have to set LAZYLLM_SENSENOVA_SECRET_KEY` togather.
#           You can apply for the API key at https://platform.sensenova.cn/home
#     * `source` needs to be specified for multiple API keys, but it does not need to be set for a single API key.

# for Windows system
# use mcp_comfig = { "command": "cmd", "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", "./"]}
mcp_comfig = {
    "command": "npx",
    "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "./"
    ]
}

mcp_client = lazyllm.tools.MCPClient(
    command_or_url=mcp_comfig["command"],
    args=mcp_comfig["args"]
)

llm = lazyllm.OnlineChatModule(source="deepseek")


def example_sync():
    agent = lazyllm.tools.agent.ReactAgent(
        llm=llm,
        tools=mcp_client.get_tools()
    )
    print("agent initialized")
    res = agent("Show me your allowed directory.")
    print(res)


async def example_async():
    agent = lazyllm.tools.agent.ReactAgent(
        llm=llm,
        tools=await mcp_client.aget_tools()
    )
    print("agent initialized")
    res = agent("Show me your allowed directory.")
    print(res)


if __name__ == '__main__':
    example_sync()
    asyncio.run(example_async())
