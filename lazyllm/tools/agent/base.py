from lazyllm.module import ModuleBase
from lazyllm import locals


class LazyLLMAgentBase(ModuleBase):
    def __init__(self, llm=None, tools=None, max_retries: int = 5, return_trace: bool = False,
                 stream: bool = False, return_last_tool_calls: bool = False, memory=None, skills=None):
        super().__init__(return_trace=return_trace)
        self._llm = llm
        self._tools = tools or []
        self._skills = skills or []
        self._memory = memory
        self._max_retries = max_retries
        self._stream = stream
        self._return_last_tool_calls = return_last_tool_calls
        self._agent = None

    def _pre_process(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            return args[0]
        return args if args else kwargs

    def _post_process(self, result):
        return result

    def build_agent(self):
        raise NotImplementedError('Subclasses must implement build_agent().')

    def _ensure_agent(self):
        if self._agent is None:
            self._agent = self.build_agent()

    def forward(self, *args, **kwargs):
        self._ensure_agent()
        pre = self._pre_process(*args, **kwargs)
        if isinstance(pre, tuple):
            result = self._agent(*pre)
        elif isinstance(pre, dict):
            result = self._agent(**pre)
        else:
            result = self._agent(pre)
        return self._post_process(result)

    def _assert_llm_tools(self, require_tools: bool = True):
        assert self._llm is not None, 'llm cannot be empty.'
        if require_tools:
            assert self._tools, 'tools cannot be empty.'

    def _pop_completed_tool_calls(self):
        if self._return_last_tool_calls and locals['_lazyllm_agent'].get('completed'):
            return locals['_lazyllm_agent'].pop('completed')
        return None

    def _pop_workspace_tool_calls(self, key: str = 'tool_call_trace'):
        if self._return_last_tool_calls and locals['_lazyllm_agent'].get('workspace', {}).get(key):
            return locals['_lazyllm_agent'].pop('workspace').pop(key)
        return None
