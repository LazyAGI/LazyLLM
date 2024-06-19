import lazyllm
from lazyllm.agent import FuncCall
from lazyllm.agent.tools import query_weather
from lazyllm.module.onlineChatModule.openaiModule import OpenAIModule

lazyllm.config.add("openai_api_key", str, "Your OpenAI API key", "OPENAI_API_KEY")
lazyllm.config.add("weather_key", str, "Your amap weather API key", "WEATHER_KEY")

if __name__ == "__main__":
    fc = FuncCall(
        llm = OpenAIModule(base_url="https://api.openai.com/v1/", model="gpt-3.5-turbo", stream=False),
        tools = [query_weather]
    )
    resp = fc(messages=[{"role": "user", "content": "海淀区今天的天气怎么样"}])
    for item in resp:
        print("item:", item)