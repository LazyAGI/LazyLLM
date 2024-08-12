import json
import pytest
import random
from typing import Literal

import lazyllm
from lazyllm import fc_register, deploy
from lazyllm.tools import FunctionCall, FunctionCallAgent
from lazyllm.launcher import cleanup

@fc_register("tool")
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
    elif 'beijing' in location.lower():
        return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register("tool")
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
        return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
    elif 'beijing' in location.lower():
        return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@pytest.fixture()
def exe_trainable_single_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    model = params.get('model', None)
    tools = params.get("tools", [])
    query = params.get("query", "")
    if not query or not tools:
        raise ValueError(f"query: {query} and {tools} cannot be empty.")
    llm = lazyllm.TrainableModule(model).deploy_method(deploy.vllm).start()

    print(f"\nStarting test 【{model}】 function calling")
    fc = FunctionCall(llm, tools)
    ret = fc(query)
    yield ret
    print(f"\n【{model}】 function calling test done.")

@pytest.fixture()
def exe_trainable_parallel_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    model = params.get('model', None)
    tools = params.get("tools", [])
    query = params.get("query", "")
    if not query or not tools:
        raise ValueError(f"query: {query} and tools: {tools} cannot be empty.")
    llm = lazyllm.TrainableModule(model).deploy_method(deploy.vllm).start()

    agent = FunctionCallAgent(llm, tools)
    print(f"\nStarting test 【{model}】parallel function calling")
    ret = agent(query)
    yield ret
    print(f"\n【{model}】parallel function calling test done.")

tools = ["get_current_weather", "get_n_day_weather_forecast"]
squery1 = "What's the weather like today in celsius in Tokyo."
squery2 = "What will the weather be like in celsius in Paris tomorrow?"
mquery1 = "What's the weather like today in celsius in Tokyo and Paris."
mquery2 = "What will the weather be like in fahrenheit in san francisco and beijing tomorrow?"
vModels = ['GLM-4-9B-Chat', 'Qwen2-7B-Instruct']

class TestTrainableFunctionCall(object):
    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        cleanup()

    @pytest.mark.parametrize("exe_trainable_single_function_call",
                             [{"model": random.choice(vModels), "tools": tools, "query": squery1},
                              {"model": random.choice(vModels), "tools": tools, "query": squery2}],
                             indirect=True)
    def test_trainable_single_function_call(self, exe_trainable_single_function_call):
        ret = exe_trainable_single_function_call
        assert isinstance(ret, list)

    @pytest.mark.parametrize("exe_trainable_parallel_function_call",
                             [{"model": random.choice(vModels), 'tools': tools, 'query': mquery1},
                              {"model": random.choice(vModels), 'tools': tools, 'query': mquery2}],
                             indirect=True)
    def test_trainable_parallel_function_call(self, exe_trainable_parallel_function_call):
        ret = exe_trainable_parallel_function_call
        assert isinstance(ret, str)
