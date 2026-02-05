import json
import pytest
from typing import Literal
import wikipedia

import lazyllm
from lazyllm import fc_register, deploy
from lazyllm.tools import FunctionCall, ReactAgent, PlanAndSolveAgent, ReWOOAgent
from lazyllm.launcher import cleanup

@fc_register('tool')
def get_current_weather(location: str,
                        unit: Literal['Fahrenheit', 'Celsius', 'fahrenheit', 'celsius', 'C', 'F'] = 'fahrenheit'):
    '''
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        unit (str): The temperature unit to use. Infer this from the users location.
    '''
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

@fc_register('tool')
def get_n_day_weather_forecast(location: str, num_days: int,
                               unit: Literal['Celsius', 'Fahrenheit', 'celsius', 'fahrenheit', 'C', 'F'] = 'fahrenheit'):
    '''
    Get an N-day weather forecast

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        num_days (int): The number of days to forecast.
        unit (Literal['Celsius', 'Fahrenheit', 'celsius', 'fahrenheit', 'C', 'F']): The temperature unit to use. Infer this from the users location.
    '''  # noqa E501
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', 'num_days': num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', 'num_days': num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', 'num_days': num_days})
    elif 'beijing' in location.lower():
        return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', 'num_days': num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register('tool')
def multiply_tool(a: int, b: int) -> int:
    '''
    Multiply two integers and return the result integer

    Args:
        a (int): multiplier
        b (int): multiplier

    Returns:
        int: result
    '''
    return a * b

@fc_register('tool')
def add_tool(a: int, b: int):
    '''
    Add two integers and returns the result integer

    Args:
        a (int): addend
        b (int): addend
    '''
    return a + b

@fc_register('tool')
def is_even_or_odd(number):
    '''
    定义一个函数，用于判断一个数字是奇数还是偶数

    Args:
        number (int): 输入数值

    Returns:
        str: 输出
    '''
    if number % 2 == 0:
        return f'{number}是偶数'
    else:
        return f'{number}是奇数'

@fc_register('tool')
def WikipediaWorker(input: str):
    '''
    Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge \
    about people, places, companies, historical events, or other subjects. The response are long and might \
    contain some irrelevant information. Input should be a search query.

    Args:
        input (str): search query.
    '''
    try:
        evidence = wikipedia.page(input).content
        evidence = evidence.split('\n\n')[0]
    except wikipedia.PageError:
        evidence = f'Could not find [{input}]. Similar: {wikipedia.search(input)}'
    except wikipedia.DisambiguationError:
        evidence = f'Could not find [{input}]. Similar: {wikipedia.search(input)}'
    return evidence

@fc_register('tool')
def LLMWorker(input: str):
    '''
    A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. \
    Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.

    Args:
        input (str): instruction
    '''
    # Simulated LLM response
    return 'I have no idea.'

@pytest.fixture()
def exe_trainable_single_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    tools = params.get('tools', [])
    query = params.get('query', '')
    llm = request.cls.llm
    if not query or not tools:
        raise ValueError(f'query: {query} and {tools} cannot be empty.')

    print(f'\nStarting test 【{llm}】 function calling')
    fc = FunctionCall(llm, tools)
    ret = fc(query)
    yield ret
    print(f'\n【{llm}】 function calling test done.')

@pytest.fixture()
def exe_trainable_parallel_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    tools = params.get('tools', [])
    query = params.get('query', '')
    llm = request.cls.llm
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')

    agent = ReactAgent(llm, tools)
    print(f'\nStarting test 【{llm}】parallel function calling')
    ret = agent(query)
    yield ret
    print(f'\n【{llm}】parallel function calling test done.')

@pytest.fixture()
def exe_trainable_advance_agent(request):
    params = request.param if hasattr(request, 'param') else {}
    tools = params.get('tools', [])
    query = params.get('query', '')
    Agent = params.get('Agent', None)
    return_last_tool_calls = params.get('return_last_tool_calls', False)
    llm = request.cls.llm
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')
    if Agent is None:
        raise ValueError(f'Agent: {Agent} must be a valid value.')

    agent = Agent(llm, tools, return_last_tool_calls=return_last_tool_calls)
    print(f'\nStarting test 【{llm}】 {Agent}.')
    ret = agent(query)
    yield ret
    print(f'\n【{llm}】 {Agent} test done.')

tools = ['get_current_weather', 'get_n_day_weather_forecast']
squery1 = 'What\'s the weather like today in celsius in Tokyo.'
squery2 = 'What will the weather be like in celsius in Paris tomorrow?'
mquery1 = 'What\'s the weather like today in celsius in Tokyo and Paris.'
mquery2 = 'What will the weather be like in fahrenheit in san francisco and beijing tomorrow?'
agentQuery = '计算 20*(45+23)*4, Calculate step by step.'
agentQuery2 = '美国历届总统就职时年龄最大的是谁'

class TestTrainableFunctionCall(object):
    @classmethod
    def setup_class(cls):
        cls.llm = lazyllm.TrainableModule('Qwen2.5-32B-Instruct').deploy_method(
            deploy.vllm).start()

    @classmethod
    def teardown_class(cls):
        cleanup()

    @pytest.mark.parametrize('exe_trainable_single_function_call',
                             [{'tools': tools, 'query': squery1},
                              {'tools': tools, 'query': squery2}],
                             indirect=True)
    def test_trainable_single_function_call(self, exe_trainable_single_function_call):
        ret = exe_trainable_single_function_call
        assert isinstance(ret, dict)

    @pytest.mark.parametrize('exe_trainable_parallel_function_call',
                             [{'tools': tools, 'query': mquery1},
                              {'tools': tools, 'query': mquery2}],
                             indirect=True)
    def test_trainable_parallel_function_call(self, exe_trainable_parallel_function_call):
        ret = exe_trainable_parallel_function_call
        assert isinstance(ret, str)

    @pytest.mark.parametrize('exe_trainable_advance_agent',
                             [{'tools': ['WikipediaWorker', 'LLMWorker'], 'query': agentQuery2, 'Agent': ReactAgent},
                              {'tools': ['multiply_tool', 'add_tool'], 'query': agentQuery, 'Agent': PlanAndSolveAgent},
                              {'tools': ['multiply_tool', 'add_tool'], 'query': agentQuery, 'Agent': ReWOOAgent}],
                             indirect=True)
    def test_trainable_advance_agent(self, exe_trainable_advance_agent):
        ret = exe_trainable_advance_agent
        assert 'retrying' not in ret

    @pytest.mark.parametrize('exe_trainable_advance_agent',
                             [{'tools': ['multiply_tool', 'add_tool'], 'query': agentQuery, 'Agent': ReactAgent,
                               'return_last_tool_calls': True},
                              {'tools': ['multiply_tool', 'add_tool'], 'query': agentQuery, 'Agent': PlanAndSolveAgent,
                               'return_last_tool_calls': True},
                              {'tools': ['multiply_tool', 'add_tool'], 'query': agentQuery, 'Agent': ReWOOAgent,
                               'return_last_tool_calls': True}],
                             indirect=True)
    def test_return_last_tool_calls(self, exe_trainable_advance_agent):
        ret = exe_trainable_advance_agent
        assert isinstance(ret, list) and len(ret) > 0
        assert 'function' in ret[0] and 'tool_call_result' in ret[0]
