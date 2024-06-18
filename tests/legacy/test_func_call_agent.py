import lazyllm
from lazyllm.agent import FuncCallAgent
from lazyllm.agent.tools import ToolManager, query_weather, web_search
from lazyllm.module.onlineChatModule.openaiModule import OpenAIModule

lazyllm.config.add('google_key', str, 'Your Google API key', 'GOOGLE_KEY')
lazyllm.config.add('google_cx', str, 'Your Google Custom Search Engine ID', 'GOOGLE_CX')
lazyllm.config.add("openai_api_key", str, "Your OpenAI API key", "OPENAI_API_KEY")
lazyllm.config.add("weather_key", str, "Your amap weather API key", "WEATHER_KEY")

tools = [query_weather, web_search]

if __name__ == '__main__':
    fc_agent = FuncCallAgent(
        llm=OpenAIModule(base_url="http://localhost:22341/v1", model='qwen2'),
        tool_manager=ToolManager(tools),
        name = 'fc_agent',
    )
    resp = fc_agent.forward(messages=[{"role": "user", "content": "海淀区今天的天气怎么样"}])
    for item in resp:
        print("item:", item)

    # resp = fc_agent.forward(messages=[{"role": "user", "content": "用谷歌搜索一下周杰伦"}])
    # for item in resp:
    #     print("item:", item)