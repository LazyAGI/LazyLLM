from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, loop, locals, Color, package, FileSystemQueue, colored_text
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable
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

    def __init__(self, llm, tools: List[Union[str, Callable]], *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None):
        super().__init__(return_trace=return_trace)

        self._tools_manager = ToolManager(tools, return_trace=return_trace)
        self._prompter = ChatPrompter(instruction=_prompt or FC_PROMPT, tools=self._tools_manager.tools_description)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter()).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.ins = StreamResponse('Received instruction:', prefix_color=Color.yellow,
                                            color=Color.green, stream=stream)
            self._impl.pre_action = self._build_history
            self._impl.llm = self._llm
            self._impl.dis = StreamResponse('Decision-making or result in this round:',
                                            prefix_color=Color.yellow, color=Color.green, stream=stream)
            self._impl.post_action = self._post_action

    def _build_history(self, input: Union[str, list]):
        history_idx = len(locals['_lazyllm_agent']['workspace'].setdefault('history', []))
        if isinstance(input, str):
            locals['_lazyllm_agent']['workspace']['history'].append({'role': 'user', 'content': input})
        elif isinstance(input, dict):
            tool_call_results = [
                {
                    'role': 'tool',
                    'content': str(tool_call['tool_call_result']),
                    'tool_call_id': tool_call['id'],
                    'name': tool_call['function']['name'],
                } for tool_call in locals['_lazyllm_agent']['workspace']['tool_call_trace']
            ]
            locals['_lazyllm_agent']['workspace']['history'].append(
                {'role': 'assistant', 'content': input.get('content', ''), 'tool_calls': input.get('tool_calls', [])}
            )
            input = {'input': tool_call_results}
            history_idx += 1
            locals['_lazyllm_agent']['workspace']['history'].extend(tool_call_results)
        locals['chat_history'][self._llm._module_id] = locals['_lazyllm_agent']['workspace']['history'][:history_idx]
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
class FunctionCallAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        self._return_last_tool_calls = return_last_tool_calls
        self._fc = FunctionCall(llm, tools, return_trace=return_trace, stream=stream)
        self._agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._fc._llm.used_by(self._module_id)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history) if llm_chat_history is not None else self._agent(query)
        if isinstance(ret, str) and self._return_last_tool_calls and locals['_lazyllm_agent'].get('completed'):
            return locals['_lazyllm_agent'].pop('completed')
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(ValueError(f'After retrying \
            {self._max_retries} times, the function call agent still fails to call successfully.'))
