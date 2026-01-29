from lazyllm.module import ModuleBase
from lazyllm import locals, LOG
from .toolsManager import ToolManager
from .skill_manager import SkillManager
from .file_tool import (  # noqa: F401
    read_file,
    list_dir,
    search_in_files,
    make_dir,
    write_file,
    delete_file,
    move_file,
)
from .shell_tool import shell_tool  # noqa: F401
from .download_tool import download_file  # noqa: F401


class LazyLLMAgentBase(ModuleBase):
    def __init__(self, llm=None, tools=None, max_retries: int = 5, return_trace: bool = False,
                 stream: bool = False, return_last_tool_calls: bool = False, use_skills: bool = False,
                 skills: list[str] = None, memory=None, desc: str = '', sub_agents=None):
        super().__init__(return_trace=return_trace)
        if skills and not use_skills:
            LOG.warning('use_skills is False but skills provided; enabling skills.')
        use_skills = use_skills or bool(skills)
        self._llm = llm
        self._tools = list(tools) if tools else []
        self._memory = memory
        self._use_skills = use_skills
        self._skills = skills or []
        self._desc = desc
        self._sub_agents = sub_agents if sub_agents is not None else []
        self._max_retries = max_retries
        self._stream = stream
        self._return_last_tool_calls = return_last_tool_calls
        self._agent = None
        self._skill_manager = None

        if self._use_skills:
            self._skill_manager = SkillManager(skills=self._skills)
            self._ensure_default_skill_tools()
        self._tools_manager = ToolManager(self._tools, return_trace=return_trace)

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

    def _ensure_default_skill_tools(self):
        defaults = [
            'read_file', 'list_dir', 'search_in_files', 'make_dir',
            'write_file', 'delete_file', 'move_file',
            'shell_tool', 'download_file',
        ]
        for name in defaults:
            if name not in self._tools:
                self._tools.append(name)
        if self._skill_manager:
            for tool in self._skill_manager.get_skill_tools():
                self._tools.append(tool)

    def _build_extra_system_prompt(self, task: str) -> str:
        if not self._use_skills or not self._skill_manager:
            return ''
        return self._skill_manager.build_prompt(task)

    @property
    def desc(self) -> str:
        return self._desc

    @desc.setter
    def desc(self, desc: str):
        self._desc = desc

    def _pop_completed_tool_calls(self):
        if self._return_last_tool_calls and locals['_lazyllm_agent'].get('completed'):
            return locals['_lazyllm_agent'].pop('completed')
        return None

    def _pop_workspace_tool_calls(self, key: str = 'tool_call_trace'):
        if self._return_last_tool_calls and locals['_lazyllm_agent'].get('workspace', {}).get(key):
            return locals['_lazyllm_agent'].pop('workspace').pop(key)
        return None
