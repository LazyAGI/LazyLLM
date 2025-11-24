import pytest
import lazyllm
from lazyllm import ReactAgent, PlanAndSolveAgent, ReWOOAgent
from lazyllm.tools import FunctionCall
import random
from ... import tools as _  # noqa F401


@pytest.fixture()
def exe_onlinechat_chat(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    query = params.get('query', '')
    if not query:
        raise ValueError(f'query: {query} cannot be empty.')
    sources = ['kimi', 'glm', 'qwen']
    if source is None or source not in sources:
        raise ValueError(f'The source {source} field must contain the value in the list {sources}')
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    print(f'\nStarting test 【{source}】chat')
    ret = llm(query, llm_chat_history=[])
    yield ret
    print(f'\n【{source}】chat test done.')

def exe_onlinechat_single_function_call(source, model, tools, query):
    if not query or not tools:
        raise ValueError(f'query: {query} and tools cannot be empty.')
    sources = ['kimi', 'glm', 'qwen']
    if source is None or source not in sources:
        raise ValueError(f'The source {source} field must contain the value in the list {sources}')
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model)
    else:
        llm = lazyllm.OnlineChatModule(source=source)

    print(f'\nStarting test 【{source}】 function calling')
    fc = FunctionCall(llm, tools)
    ret = fc(query, [])
    print(f'\n【{source}】 function calling test done.')
    return ret

@pytest.fixture()
def exe_onlinechat_parallel_function_call(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get('tools', [])
    query = params.get('query', '')
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')
    sources = ['kimi', 'glm', 'qwen']
    if source is None or source not in sources:
        raise ValueError(f'The source {source} field must contain the value in the list {sources}')
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    agent = ReactAgent(llm, tools)
    print(f'\nStarting test 【{source}】parallel function calling')
    ret = agent(query, [])
    yield ret
    print(f'\n【{source}】parallel function calling test done.')

@pytest.fixture()
def exe_onlinechat_advance_agent(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get('tools', [])
    query = params.get('query', '')
    Agent = params.get('Agent', None)
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')
    if Agent is None:
        raise ValueError(f'Agent: {Agent} must be a valid value.')
    sources = ['kimi', 'glm', 'qwen']
    if source is None or source not in sources:
        raise ValueError(f'The source {source} field must contain the value in the list {sources}')
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    agent = Agent(llm, tools)
    print(f'agent: {agent}')
    print(f'\nStarting test 【{source}】 {Agent}.')
    ret = agent(query)
    print(f'ret: {ret}')
    yield ret
    print(f'\n 【{source}】{Agent} test done.')


tools = ['get_current_weather', 'get_n_day_weather_forecast']
squery = 'What\'s the weather like today in celsius in Tokyo.'
mquery = 'What\'s the weather like today in celsius in Tokyo and Paris.'
agentQuery = 'What is 20+(2*4)? Calculate step by step '
models = ['kimi', 'glm', 'qwen']
agentQuery2 = 'What is the name of the cognac house that makes the main ingredient in The Hennchata?'

@pytest.mark.skip_on_linux
class TestOnlineChatFunctionCall(object):
    @pytest.mark.parametrize('exe_onlinechat_chat',
                             [{'source': 'qwen', 'query': squery}],
                             indirect=True)
    def test_onlinechat_chat(self, exe_onlinechat_chat):
        ret = exe_onlinechat_chat
        assert isinstance(ret, str, )

    @pytest.mark.ignore_cache_on_change('lazyllm/module/llms/onlinemodule/supplier/glm.py')
    def test_onlinechat_single_function_call_glm(self):
        ret = exe_onlinechat_single_function_call('glm', 'GLM-4-Flash', tools, squery)
        assert isinstance(ret, dict)

    @pytest.mark.ignore_cache_on_change('lazyllm/module/llms/onlinemodule/supplier/qwen.py')
    def test_onlinechat_single_function_call_qwen(self):
        ret = exe_onlinechat_single_function_call('qwen', 'qwen-turbo', tools, squery)
        assert isinstance(ret, dict)

    @pytest.mark.parametrize('exe_onlinechat_parallel_function_call',
                             [{'source': 'kimi', 'tools': tools, 'query': mquery}],
                             indirect=True)
    def test_onlinechat_parallel_function_call(self, exe_onlinechat_parallel_function_call):
        ret = exe_onlinechat_parallel_function_call
        assert isinstance(ret, str)

    @pytest.mark.parametrize('exe_onlinechat_advance_agent',
                             [{'source': random.choice(models), 'tools': ['WikipediaWorker', 'LLMWorker'],
                               'query': agentQuery2, 'Agent': ReactAgent},
                              {'source': random.choice(models), 'tools': ['multiply_tool', 'add_tool'],
                               'query': agentQuery, 'Agent': PlanAndSolveAgent},
                              {'source': random.choice(models), 'tools': ['multiply_tool', 'add_tool'],
                               'query': agentQuery, 'Agent': ReWOOAgent}],
                             indirect=True)
    def test_onlinechat_advance_agent(self, exe_onlinechat_advance_agent):
        ret = exe_onlinechat_advance_agent
        assert 'retrying' not in ret
