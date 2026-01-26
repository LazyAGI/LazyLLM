import pytest
import lazyllm
from lazyllm import ReactAgent, PlanAndSolveAgent, ReWOOAgent
from lazyllm.tools import FunctionCall
from ... import tools as _  # noqa F401


pytestmark = pytest.mark.advanced_test

DEFAULT_SOURCE = 'qwen'
DEFAULT_MODEL = 'qwen-turbo'

@pytest.fixture()
def exe_onlinechat_chat(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', DEFAULT_SOURCE)
    model = params.get('model', None)
    stream = params.get('stream', False)
    query = params.get('query', '')
    if not query:
        raise ValueError(f'query: {query} cannot be empty.')
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
    source = params.get('source', DEFAULT_SOURCE)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get('tools', [])
    query = params.get('query', '')
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')
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
    source = params.get('source', DEFAULT_SOURCE)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get('tools', [])
    query = params.get('query', '')
    Agent = params.get('Agent', None)
    if not query or not tools:
        raise ValueError(f'query: {query} and tools: {tools} cannot be empty.')
    if Agent is None:
        raise ValueError(f'Agent: {Agent} must be a valid value.')
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
agentQuery2 = 'What is the name of the cognac house that makes the main ingredient in The Hennchata?'

@pytest.mark.skip_on_linux
class TestOnlineChatFunctionCall(object):
    @pytest.mark.parametrize('exe_onlinechat_chat',
                             [{'source': DEFAULT_SOURCE, 'query': squery}],
                             indirect=True)
    def test_onlinechat_chat(self, exe_onlinechat_chat):
        ret = exe_onlinechat_chat
        assert isinstance(ret, str, )

    def test_onlinechat_single_function_call_glm(self):
        ret = exe_onlinechat_single_function_call(DEFAULT_SOURCE, DEFAULT_MODEL, tools, squery)
        assert isinstance(ret, dict)

    def test_onlinechat_single_function_call_qwen(self):
        ret = exe_onlinechat_single_function_call(DEFAULT_SOURCE, DEFAULT_MODEL, tools, squery)
        assert isinstance(ret, dict)

    @pytest.mark.parametrize('exe_onlinechat_parallel_function_call',
                             [{'source': DEFAULT_SOURCE, 'tools': tools, 'query': mquery}],
                             indirect=True)
    def test_onlinechat_parallel_function_call(self, exe_onlinechat_parallel_function_call):
        ret = exe_onlinechat_parallel_function_call
        assert isinstance(ret, str)

    @pytest.mark.parametrize('exe_onlinechat_advance_agent',
                             [{'source': DEFAULT_SOURCE, 'tools': ['WikipediaWorker', 'LLMWorker'],
                               'query': agentQuery2, 'Agent': ReactAgent},
                              {'source': DEFAULT_SOURCE, 'tools': ['multiply_tool', 'add_tool'],
                               'query': agentQuery, 'Agent': PlanAndSolveAgent},
                              {'source': DEFAULT_SOURCE, 'tools': ['multiply_tool', 'add_tool'],
                               'query': agentQuery, 'Agent': ReWOOAgent}],
                             indirect=True)
    def test_onlinechat_advance_agent(self, exe_onlinechat_advance_agent):
        ret = exe_onlinechat_advance_agent
        assert 'retrying' not in ret
