import lazyllm
from lazyllm.agent import FuncCallAgent
from lazyllm.module.onlineChatModule.openaiModule import OpenAIModule

lazyllm.config.add("openai_api_key", str, "Your OpenAI API key", "OPENAI_API_KEY")

if __name__ == '__main__':
    fc_agent = FuncCallAgent(
        llm=OpenAIModule(base_url="http://localhost:22341/v1", model='chatgpt-3.5-turbo'),
        tools_list=['query_weather'],
        name = 'fc_agent',
    )
    resp = fc_agent.forward("海淀区今天的天气怎么样")
    for item in resp:
        print("item:", item)