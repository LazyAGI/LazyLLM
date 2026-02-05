import os
from typing import Iterable, Optional, Union

import lazyllm
from lazyllm.module import ModuleBase
from lazyllm import locals, once_wrapper
from .toolsManager import ToolManager
from .skill_manager import SkillManager, SKILLS_PROMPT
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
                 stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Optional[Union[bool, str, Iterable[str]]] = None, memory=None,
                 desc: str = '', workspace: Optional[str] = None):
        super().__init__(return_trace=return_trace)
        use_skills, skills = self._normalize_skills_config(skills)
        self._llm = llm
        self._tools = list(tools) if tools else []
        self._memory = memory
        self._skills = skills or []
        self._desc = desc
        self._max_retries = max_retries
        self._stream = stream
        self._return_last_tool_calls = return_last_tool_calls
        self._workspace = self._init_workspace(workspace)
        self._agent = None
        self._skill_manager = None

        if use_skills:
            self._skill_manager = SkillManager(skills=self._skills)
            self._ensure_default_skill_tools()
        self._tools_manager = ToolManager(self._tools, return_trace=return_trace)

    @staticmethod
    def _normalize_skills_config(skills: Optional[Union[bool, str, Iterable[str]]]):
        if isinstance(skills, bool):
            return skills, []
        if skills is None:
            return False, []
        if isinstance(skills, str):
            if not skills.strip():
                return False, []
            return True, [skills]
        if isinstance(skills, (list, tuple, set)):
            normalized = [item for item in skills if item]
            if not normalized:
                return False, []
            return True, normalized
        raise TypeError('skills must be a bool, str, or list of skill names.')

    def _init_workspace(self, workspace: Optional[str]) -> str:
        path = workspace or os.path.join(lazyllm.config['home'], 'agent_workspace')
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
        return path

    def _pre_process(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            return args[0]
        return args if args else kwargs

    def _post_process(self, result):
        return result

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        raise NotImplementedError('Subclasses must implement build_agent().')

    def forward(self, *args, **kwargs):
        self.build_agent()
        if self._agent is None:
            raise RuntimeError('build_agent() did not initialize _agent.')
        pre = self._pre_process(*args, **kwargs)
        if isinstance(pre, tuple):
            result = self._agent(*pre)
        elif isinstance(pre, dict):
            result = self._agent(**pre)
        else:
            result = self._agent(pre)
        return self._post_process(result)

    def _assert_tools(self):
        assert self._tools, 'tools cannot be empty.'

    def _ensure_default_skill_tools(self):
        builtin_group = getattr(lazyllm, 'builtin_tools', None)
        builtin_names = []
        if builtin_group:
            builtin_names = [f'builtin_tools.{name}' for name in builtin_group.keys()]
        existing = set()
        for tool in self._tools:
            if isinstance(tool, str):
                existing.add(tool.split('.')[-1])
            elif hasattr(tool, '__name__'):
                existing.add(tool.__name__)
        for name in builtin_names:
            if name.split('.')[-1] not in existing:
                self._tools.append(name)
        if self._skill_manager:
            for tool in self._skill_manager.get_skill_tools():
                self._tools.append(tool)

    def _append_skills_prompt(self, prompt: str) -> str:
        if not self._skill_manager:
            return prompt
        return f'{prompt}\n\n{SKILLS_PROMPT}'

    def _wrap_user_input_with_skills(self, query: str):
        if not self._skill_manager:
            return query
        return self._skill_manager.wrap_input(query, query)

    def _append_workspace_prompt(self, prompt: str) -> str:
        return (
            f'{prompt}\n\n## Workspace\n'
            f'- Default workspace: `{self._workspace}`\n'
            '- Prefer creating and updating files under this workspace.\n'
            '- Use absolute paths under this workspace when possible.\n'
        )

    @property
    def workspace(self) -> str:
        return self._workspace

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
