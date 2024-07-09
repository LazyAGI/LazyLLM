import json
import pytest
import lazyllm
from lazyllm import fc_register, pipeline, package, loop
from lazyllm.tools import FunctionCall
from typing import Literal

fc_register.new_group("functionCall")

@fc_register("functionCall")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
    """
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        unit (str): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register("functionCall")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
    """
    Get an N-day weather forecast

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        num_days (int): The number of days to forecast.
        unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@pytest.fixture()
def exe_onlineChat_Chat(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    query = params.get("query", "")
    if not query:
        raise ValueError(f"query: {query} cannot be empty.")
    sources = ["kimi", "glm", "qwen", "sensenova"]
    if source is None or source not in sources:
        raise ValueError(f"The source {source} field must contain the value in the list {sources}")
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream).field_extractor(".")
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream).field_extractor(".")

    print(f"\nStarting test 【{source}】chat")
    ret = llm(query, [])
    role = ret['role']
    content = ret['content']
    yield (role, content)
    print(f"\n【{source}】chat test done.")

@pytest.fixture()
def exe_onlineChat_single_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get("tools", [])
    query = params.get("query", "")
    if not query or not tools:
        raise ValueError(f"query: {query} and tools cannot be empty.")
    sources = ["kimi", "glm", "qwen", "sensenova"]
    if source is None or source not in sources:
        raise ValueError(f"The source {source} field must contain the value in the list {sources}")
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream).field_extractor(".")
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream).field_extractor(".")

    print(f"\nStarting test 【{source}】 function calling")
    fc = FunctionCall(llm, tools)
    ret = fc(query, [])
    input = ret[1]['input']
    content = json.loads(input['content'])
    tool_name = input['name']
    loc = content['location']
    role = input['role']
    temperature = content['temperature']
    unit = content['unit']
    yield (role, tool_name, loc, temperature, unit)
    print(f"\n【{source}】 function calling test done.")

@pytest.fixture()
def exe_onlineChat_parallel_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get("tools", [])
    query = params.get("query", "")
    if not query or not tools:
        raise ValueError(f"query: {query} and tools: {tools} cannot be empty.")
    sources = ["kimi", "glm", "qwen", "sensenova"]
    if source is None or source not in sources:
        raise ValueError(f"The source {source} field must contain the value in the list {sources}")
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream).field_extractor(".")
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream).field_extractor(".")

    with pipeline() as agent:
        with loop(stop_condition=lambda x, y, z: x, count=5) as agent.lp:
            with pipeline() as agent.lp.ppl:
                agent.lp.ppl.m1 = lambda isFinish, input, history: package(input, history)
                agent.lp.ppl.m2 = FunctionCall(llm, tools)
        agent.post_action = lambda isFinish, output, history: output

    print(f"\nStarting test 【{source}】parallel function calling")
    ret = agent(False, query, [])
    yield ret
    print(f"\n【{source}】parallel function calling test done.")

tools = ["get_current_weather", "get_n_day_weather_forecast"]
squery = "What's the weather like today in Tokyo."
mquery = "What's the weather like today in Tokyo and Paris."

class TestOnlineChatFunctionCall(object):
    @pytest.mark.parametrize("exe_onlineChat_Chat",
                             [{'source': 'sensenova', 'model': 'SenseChat-Turbo', 'query': squery}],
                             indirect=True)
    def test_onlineChat_chat(self, exe_onlineChat_Chat):
        ret = exe_onlineChat_Chat
        assert len(ret) == 2
        assert ret[0] == 'assistant' and len(ret[1]) > 10

    @pytest.mark.parametrize("exe_onlineChat_single_function_call",
                             [{'source': 'glm', "model": "GLM-4-Flash", "tools": tools, "query": squery},
                              {'source': 'qwen', "model": "qwen-turbo", "tools": tools, "query": squery}],
                             indirect=True)
    def test_onlineChat_single_function_call(self, exe_onlineChat_single_function_call):
        ret = exe_onlineChat_single_function_call
        assert len(ret) == 5
        assert ret == ('tool', 'get_current_weather', 'Tokyo', '10', 'celsius')

    @pytest.mark.parametrize("exe_onlineChat_parallel_function_call",
                             [{'source': 'kimi', 'tools': tools, 'query': mquery}],
                             indirect=True)
    def test_onlineChat_parallel_function_call(self, exe_onlineChat_parallel_function_call):
        ret = exe_onlineChat_parallel_function_call
        assert len(ret) > 10 and "Tokyo" in ret and "Paris" in ret and "10" in ret and "22" in ret
