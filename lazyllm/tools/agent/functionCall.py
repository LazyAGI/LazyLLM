from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, loop, locals, Color, package, FileSystemQueue, colored_text, once_wrapper
from .toolsManager import ToolManager
from .skill_manager import SKILLS_PROMPT
from typing import List, Any, Dict, Union, Callable, Optional
from .base import LazyLLMAgentBase
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.common.deprecated import deprecated
import re
import json

FC_PROMPT = f'''# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs.
{FC_PROMPT_PLACEHOLDER}

Don\'t make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.\n
'''


class StreamResponse():
    def __init__(self, prefix: str, prefix_color: str = None, color: str = None, stream: bool = False):
        self.stream = stream
        self.prefix = prefix
        self.prefix_color = prefix_color
        self.color = color

    def __call__(self, *inputs):
        if self.stream: FileSystemQueue().enqueue(colored_text(f'\n{self.prefix}\n', self.prefix_color))
        if len(inputs) == 1:
            if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs[0]}', self.color))
            return inputs[0]
        if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs}', self.color))
        return package(*inputs)


class FunctionCall(ModuleBase):

    def __init__(self, llm, tools: Optional[List[Union[str, Callable]]] = None, *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None, _tool_manager: Optional[ToolManager] = None,
                 skill_manager=None, workspace: Optional[str] = None):
        super().__init__(return_trace=return_trace)
        if _tool_manager is None:
            assert tools, 'tools cannot be empty.'
            self._tools_manager = ToolManager(tools, return_trace=return_trace)
        else:
            self._tools_manager = _tool_manager
        self._skill_manager = skill_manager
        self._workspace = workspace
        prompt = _prompt or FC_PROMPT
        if self._workspace:
            prompt = (
                f'{prompt}\n\n## Workspace\n'
                f'- Default workspace: `{self._workspace}`\n'
                '- Prefer creating/updating files under this workspace.\n'
                '- Use absolute paths under this workspace when creating files.\n'
            )
        if self._skill_manager:
            prompt = f'{prompt}\n\n{SKILLS_PROMPT}'
            self._prompter = ChatPrompter(
                instruction=prompt,
                tools=self._tools_manager.tools_description,
                extra_keys=['available_skills']
            )
        else:
            self._prompter = ChatPrompter(instruction=prompt, tools=self._tools_manager.tools_description)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter()).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.ins = StreamResponse('Received instruction:', prefix_color=Color.yellow,
                                            color=Color.green, stream=stream)
            self._impl.pre_action = self._build_history
            self._impl.llm = self._llm
            self._impl.dis = StreamResponse('Decision-making or result in this round:',
                                            prefix_color=Color.yellow, color=Color.green, stream=stream)
            self._impl.post_action = self._post_action

    def _build_history(self, input: Union[str, dict, list]):
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        if self._skill_manager and isinstance(input, dict) and input.get('available_skills'):
            workspace['available_skills'] = input['available_skills']
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
            workspace['history'].append(
                {'role': 'assistant', 'content': input.get('content', ''), 'tool_calls': input.get('tool_calls', [])}
            )
            input = {'input': tool_call_results}
            history_idx += 1
            workspace['history'].extend(tool_call_results)
        chat_history = workspace['history'][:history_idx]
        locals['chat_history'][self._llm._module_id] = chat_history
        if self._skill_manager and isinstance(input, dict) and 'available_skills' not in input:
            available = workspace.get('available_skills')
            if available: input['available_skills'] = available
        return input

    def _post_action(self, llm_output: Dict[str, Any]):
        if not llm_output.get('tool_calls'):
            if (match := re.search(r'Action:\s*Call\s+(\w+)\s+with\s+parameters\s+(\{.*?\})', llm_output['content'])):
                try:
                    llm_output['tool_calls'] = [{'function': {'name': match.group(1),
                                                              'arguments': json.loads(match.group(2))}}]
                except Exception: pass
        if tool_calls := llm_output.get('tool_calls'):
            if isinstance(tool_calls, list): [item.pop('index', None) for item in tool_calls]
            tool_calls_results = self._tools_manager(tool_calls)
            locals['_lazyllm_agent']['workspace']['tool_call_trace'] = [
                {**tool_call, 'tool_call_result': tool_result}
                for tool_call, tool_result in zip(tool_calls, tool_calls_results)
            ]
        else:
            llm_output = llm_output['content']
        return llm_output

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if 'workspace' not in locals['_lazyllm_agent']:
            locals['_lazyllm_agent']['workspace'] = dict(history=llm_chat_history or [])
        result = self._impl(input)

        # If the model decides not to call any tools, the result is a string. For debugging and subsequent tasks,
        # the last non-empty tool call trace is stored in locals['_lazyllm_agent']['completed'].
        if isinstance(result, str):
            locals['_lazyllm_agent']['completed'] = locals['_lazyllm_agent'].pop('workspace')\
                .pop('tool_call_trace', locals['_lazyllm_agent'].get('completed', []))
            locals['chat_history'][self._llm._module_id] = []
        return result

@deprecated('ReactAgent')
class FunctionCallAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace)
        prompt = FC_PROMPT
        self._fc = FunctionCall(self._llm, self._tools, return_trace=return_trace, stream=stream,
                                _prompt=prompt, _tool_manager=self._tools_manager,
                                skill_manager=self._skill_manager, workspace=self.workspace)
        self._fc._llm.used_by(self._module_id)

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        query = self._wrap_user_input_with_skills(query)
        if llm_chat_history is not None:
            return (query, llm_chat_history)
        return query

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_completed_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the function call agent still fails to call '
                         f'successfully.')
