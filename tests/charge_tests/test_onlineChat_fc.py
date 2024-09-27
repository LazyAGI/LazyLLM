import os
import pytest
import lazyllm
from lazyllm import ReactAgent, PlanAndSolveAgent, ReWOOAgent
from lazyllm.tools import FunctionCall, FunctionCallAgent
import random
from . import tools as _  # noqa F401


@pytest.fixture()
def exe_onlinechat_chat(request):
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
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    print(f"\nStarting test 【{source}】chat")
    ret = llm(query, llm_chat_history=[])
    yield ret
    print(f"\n【{source}】chat test done.")

@pytest.fixture()
def exe_onlinechat_single_function_call(request):
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
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    print(f"\nStarting test 【{source}】 function calling")
    fc = FunctionCall(llm, tools)
    ret = fc(query, [])
    yield ret
    print(f"\n【{source}】 function calling test done.")

@pytest.fixture()
def exe_onlinechat_parallel_function_call(request):
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
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    agent = FunctionCallAgent(llm, tools)
    print(f"\nStarting test 【{source}】parallel function calling")
    ret = agent(query, [])
    yield ret
    print(f"\n【{source}】parallel function calling test done.")

@pytest.fixture()
def exe_onlinechat_advance_agent(request):
    params = request.param if hasattr(request, 'param') else {}
    source = params.get('source', None)
    model = params.get('model', None)
    stream = params.get('stream', False)
    tools = params.get('tools', [])
    query = params.get('query', "")
    Agent = params.get('Agent', None)
    if not query or not tools:
        raise ValueError(f"query: {query} and tools: {tools} cannot be empty.")
    if Agent is None:
        raise ValueError(f"Agent: {Agent} must be a valid value.")
    sources = ["kimi", "glm", "qwen", "sensenova"]
    if source is None or source not in sources:
        raise ValueError(f"The source {source} field must contain the value in the list {sources}")
    if model:
        llm = lazyllm.OnlineChatModule(source=source, model=model, stream=stream)
    else:
        llm = lazyllm.OnlineChatModule(source=source, stream=stream)

    agent = Agent(llm, tools)
    print(f"agent: {agent}")
    print(f"\nStarting test 【{source}】 {Agent}.")
    ret = agent(query)
    print(f"ret: {ret}")
    yield ret
    print(f"\n 【{source}】{Agent} test done.")


tools = ["get_current_weather", "get_n_day_weather_forecast"]
squery = "What's the weather like today in celsius in Tokyo."
mquery = "What's the weather like today in celsius in Tokyo and Paris."
agentQuery = "What is 20+(2*4)? Calculate step by step "
models = ["kimi", "glm", "qwen", "sensenova"]
rewooquery = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"

class TestOnlineChatFunctionCall(object):
    def setup_class(self):
        self.https_proxy_bak = os.environ.get("https_proxy", '')
        self.http_proxy_bak = os.environ.get("http_proxy", '')
        os.environ['https_proxy'] = lazyllm.config['https_proxy']
        os.environ['http_proxy'] = lazyllm.config['https_proxy']

    def teardown_class(self):
        os.environ['https_proxy'] = self.https_proxy_bak
        os.environ['http_proxy'] = self.http_proxy_bak

    @pytest.mark.parametrize("exe_onlinechat_chat",
                             [{'source': 'sensenova', 'model': 'SenseChat-Turbo', 'query': squery}],
                             indirect=True)
    def test_onlinechat_chat(self, exe_onlinechat_chat):
        ret = exe_onlinechat_chat
        assert isinstance(ret, str, )

    @pytest.mark.parametrize("exe_onlinechat_single_function_call",
                             [{'source': 'glm', "model": "GLM-4-Flash", "tools": tools, "query": squery},
                              {'source': 'qwen', "model": "qwen-turbo", "tools": tools, "query": squery}],
                             indirect=True)
    def test_onlinechat_single_function_call(self, exe_onlinechat_single_function_call):
        ret = exe_onlinechat_single_function_call
        assert isinstance(ret, list)

    @pytest.mark.parametrize("exe_onlinechat_parallel_function_call",
                             [{'source': 'kimi', 'tools': tools, 'query': mquery}],
                             indirect=True)
    def test_onlinechat_parallel_function_call(self, exe_onlinechat_parallel_function_call):
        ret = exe_onlinechat_parallel_function_call
        assert isinstance(ret, str)

    @pytest.mark.parametrize("exe_onlinechat_advance_agent",
                             [{'source': random.choice(models), 'tools': ['WikipediaWorker', 'LLMWorker'],
                               "query": rewooquery, "Agent": ReactAgent},
                              {'source': random.choice(models), 'tools': ['multiply_tool', 'add_tool'],
                               "query": agentQuery, "Agent": PlanAndSolveAgent},
                              {'source': random.choice(models), 'tools': ['WikipediaWorker', 'LLMWorker'],
                               "query": rewooquery, "Agent": ReWOOAgent}],
                             indirect=True)
    def test_onlinechat_advance_agent(self, exe_onlinechat_advance_agent):
        ret = exe_onlinechat_advance_agent
        assert "retrying" not in ret
