import os
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import lazyllm
from lazyllm.module import ModuleBase
from lazyllm import locals
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, create_sandbox
from .events import (
    AgentEvent,
    TOOL_CALLS,
    TOOL_RESULTS,
)
from .stream_runner import StreamRunner
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

FC_PROMPT = f'''# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs.
{FC_PROMPT_PLACEHOLDER}

Don\'t make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.\n
'''

_COMPACTION_TRUNCATE_LEN = 200


def _compact_chat_history(history: List[Dict[str, Any]], keep_full_turns: int) -> List[Dict[str, Any]]:
    tool_indices = [i for i, m in enumerate(history) if m.get('role') == 'tool']
    cutoff = len(tool_indices) - keep_full_turns
    if cutoff <= 0:
        return list(history)
    to_truncate = set(tool_indices[:cutoff])
    result = []
    for i, msg in enumerate(history):
        if i in to_truncate:
            content = msg.get('content', '')
            if content is None:
                content = ''
            if isinstance(content, list):
                content = ' '.join(
                    p.get('text', '') if isinstance(p, dict) else str(p) for p in content
                )
            if isinstance(content, str) and len(content) > _COMPACTION_TRUNCATE_LEN:
                truncated = content[:_COMPACTION_TRUNCATE_LEN]
                msg = dict(msg, content=f'[truncated {len(content)} chars] {truncated}...')
        result.append(msg)
    return result


class LazyLLMAgentBase(ModuleBase):
    def __init__(self, llm=None, tools=None, max_retries: int = 5, return_trace: bool = False,
                 stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Optional[Union[bool, str, Iterable[str]]] = None, memory=None,
                 desc: str = '', workspace: Optional[str] = None,
                 sandbox: Union[str, LazyLLMSandboxBase, None] = 'auto',
                 fs: Optional[Any] = None, skills_dir: Optional[str] = None,
                 enable_builtin_tools: bool = True, keep_full_turns: int = 0):
        super().__init__(return_trace=return_trace)
        use_skills, skills = self._normalize_skills_config(skills)
        if not use_skills and (fs is not None or skills_dir is not None):
            import warnings
            warnings.warn(
                'fs and skills_dir are ignored because skills is not enabled. '
                'Pass skills=True (or a list of skill names) to enable skill loading.',
                UserWarning, stacklevel=2,
            )
        self._llm = llm
        self._tools = list(tools) if tools else []
        self._memory = memory
        self._skills = skills or []
        self._desc = desc
        self._max_retries = max_retries
        self._stream = stream
        self._return_last_tool_calls = return_last_tool_calls
        self._workspace = self._init_workspace(workspace)
        self._skill_manager = None
        self._sandbox = create_sandbox() if sandbox == 'auto' else sandbox
        self._builtin_tool_names = set()
        self._skill_tool_names = set()
        self._enable_builtin_tools = enable_builtin_tools
        self._keep_full_turns = keep_full_turns
        self._tool_llm = None

        if self._enable_builtin_tools:
            self._ensure_builtin_tools()
        if use_skills:
            self._skill_manager = SkillManager(dir=skills_dir, skills=self._skills, fs=fs)
            self._ensure_default_skill_tools()
        self._tools_manager = ToolManager(self._tools, return_trace=return_trace, sandbox=self._sandbox)

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

    def _make_event(self, event_type: str, **kwargs) -> AgentEvent:
        return AgentEvent(type=event_type, agent=self.__class__.__name__, **kwargs)

    @staticmethod
    def _normalize_tool_results(tool_calls, tool_calls_results):
        return [{
            'id': tool_call.get('id'),
            'name': tool_call.get('function', {}).get('name'),
            'arguments': tool_call.get('function', {}).get('arguments'),
            'result': tool_result,
        } for tool_call, tool_result in zip(tool_calls, tool_calls_results)]

    def _ensure_agent_state(self, llm_chat_history=None):
        if 'workspace' not in locals['_lazyllm_agent']:
            locals['_lazyllm_agent']['workspace'] = dict(history=llm_chat_history or [])

    def _init_tool_llm(self, prompt=None, llm=None):
        if self._tool_llm is not None:
            return
        prompt = prompt or FC_PROMPT
        llm = llm or self._llm
        self._tool_prompter = ChatPrompter(
            instruction={'system': prompt, 'user': ''},
            tools=lambda: self._tools_manager.tools_description,
            skills=self._skill_manager.build_prompt() if self._skill_manager else '',
        )
        self._tool_llm = llm.share(
            prompt=self._tool_prompter,
            format=FunctionCallFormatter(),
            stream=self._stream,
        ).used_by(self._module_id)

    def _build_history(self, input: Union[str, dict, list]):
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        if isinstance(input, str):
            workspace['history'].append({'role': 'user', 'content': input})
        elif isinstance(input, dict) and 'input' in input:
            workspace['history'].append(
                {'role': 'user', 'content': input.get('input', '')}
            )
        elif isinstance(input, dict) and input.get('role') == 'user':
            workspace['history'].append(
                {'role': 'user', 'content': input.get('content', '')}
            )
        elif isinstance(input, dict):
            tool_call_results = [
                {
                    'role': 'tool',
                    'content': str(tool_call['tool_call_result']),
                    'tool_call_id': tool_call['id'],
                    'name': tool_call['function']['name'],
                } for tool_call in workspace['tool_call_trace']
            ]
            workspace['history'].append({
                'role': 'assistant',
                'content': input.get('content', ''),
                'tool_calls': input.get('tool_calls', []),
                'reasoning_content': input.get('reasoning_content', ''),
            })
            input = {'input': tool_call_results}
            history_idx += 1
            workspace['history'].extend(tool_call_results)
        chat_history = workspace['history'][:history_idx]
        if self._keep_full_turns > 0:
            chat_history = _compact_chat_history(chat_history, self._keep_full_turns)
        locals['chat_history'][self._tool_llm._module_id] = chat_history
        return input

    def _post_action(self, llm_output: Dict[str, Any],
                     event_callback: Optional[Callable[[AgentEvent], None]] = None):
        if not llm_output.get('tool_calls'):
            if (match := re.search(r'Action:\s*Call\s+(\w+)\s+with\s+parameters\s+(\{.*?\})',
                                   llm_output['content'])):
                try:
                    llm_output['tool_calls'] = [{'function': {'name': match.group(1),
                                                              'arguments': json.loads(match.group(2))}}]
                except Exception:
                    pass
        if tool_calls := llm_output.get('tool_calls'):
            if isinstance(tool_calls, list):
                [item.pop('index', None) for item in tool_calls]
            if event_callback:
                event_callback(AgentEvent(type=TOOL_CALLS, tool_calls=tool_calls,
                                          agent=self.__class__.__name__))
            tool_calls_results = self._tools_manager(tool_calls)
            if event_callback:
                event_callback(AgentEvent(
                    type=TOOL_RESULTS,
                    agent=self.__class__.__name__,
                    tool_results=self._normalize_tool_results(tool_calls, tool_calls_results),
                ))
            locals['_lazyllm_agent']['workspace']['tool_call_trace'] = [
                {**tool_call, 'tool_call_result': tool_result}
                for tool_call, tool_result in zip(tool_calls, tool_calls_results)
            ]
        else:
            llm_output = llm_output['content']
        return llm_output

    def _run_tool_round(self, input, llm_chat_history=None,
                        event_callback: Optional[Callable[[AgentEvent], None]] = None):
        self._ensure_agent_state(llm_chat_history=llm_chat_history)
        prepared = self._build_history(input)
        llm_output = self._tool_llm(prepared)
        return self._post_action(llm_output, event_callback=event_callback)

    def _finalize_tool_result(self, result):
        if isinstance(result, str):
            workspace = locals['_lazyllm_agent'].pop('workspace', {})
            locals['_lazyllm_agent']['completed'] = workspace.pop(
                'tool_call_trace', locals['_lazyllm_agent'].get('completed', []))
            locals['_lazyllm_agent']['history'] = workspace.pop('history', [])
            locals['chat_history'][self._tool_llm._module_id] = []
        return result

    def _execute(self, input, callback=None):
        """Subclasses implement their agent logic. callback(event) for streaming events."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        pre = self._pre_process(*args, **kwargs)
        if self._stream:
            runner = StreamRunner(self.__class__.__name__)
            runner.start(lambda emit: self._execute(pre, callback=emit))
            return runner
        return self._execute(pre, callback=None)

    def _assert_tools(self):
        assert self._tools, 'tools cannot be empty.'

    def _ensure_builtin_tools(self):
        builtin_keys = []
        builtin_group = getattr(lazyllm, 'builtin_tools', None)
        if builtin_group:
            builtin_keys = list(builtin_group.keys())
        self._builtin_tool_names = {
            key[:-len('builtin_tools')] if key.endswith('builtin_tools') else key
            for key in builtin_keys
        }
        existing = set()
        for tool in self._tools:
            if isinstance(tool, str):
                existing.add(tool.split('.')[-1])
            elif hasattr(tool, '__name__'):
                existing.add(tool.__name__)
        for key in builtin_keys:
            name = key[:-len('builtin_tools')] if key.endswith('builtin_tools') else key
            if name not in existing:
                self._tools.append(f'builtin_tools.{key}')

    def _ensure_default_skill_tools(self):
        if self._skill_manager:
            for tool in self._skill_manager.get_skill_tools():
                self._skill_tool_names.add(tool.__name__)
                self._tools.append(tool)

    def _append_workspace_prompt(self, prompt: str) -> str:
        if not self._enable_builtin_tools:
            return prompt
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
    def sandbox(self) -> Optional[LazyLLMSandboxBase]:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: Optional[LazyLLMSandboxBase]):
        self._sandbox = sandbox
        if hasattr(self, '_tools_manager') and self._tools_manager is not None:
            self._tools_manager.sandbox = sandbox

    @property
    def desc(self) -> str:
        return self._desc

    @desc.setter
    def desc(self, desc: str):
        self._desc = desc

    def _pop_tool_calls(self, key: str = 'tool_call_trace'):
        if not self._return_last_tool_calls:
            return None
        if locals['_lazyllm_agent'].get('workspace', {}).get(key):
            return locals['_lazyllm_agent'].pop('workspace').pop(key)
        if locals['_lazyllm_agent'].get('completed'):
            return locals['_lazyllm_agent'].pop('completed')
        return None
