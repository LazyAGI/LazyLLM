from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, loop, globals, locals, Color, package, FileSystemQueue, colored_text
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER

FC_PROMPT = f'''# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs.
{FC_PROMPT_PLACEHOLDER}

Don\'t make assumptions about what values to plug into functions.
Ask for clarification if a user request is ambiguous.\n
'''


def function_call_hook(input: Union[str, Dict[str, Any]], history: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                       label: Any):
    if isinstance(input, dict):
        if 'query' in locals['_lazyllm_agent']['workspace']:
            history.append({'role': 'user', 'content': locals['_lazyllm_agent']['workspace'].pop('query')})
        if 'tool_call_results' in locals['_lazyllm_agent']['workspace']:
            history.extend(locals['_lazyllm_agent']['workspace'].pop('tool_call_results'))
        history.append({'role': 'assistant', 'content': input['content'], 'tool_calls': input['tool_calls']})

        tool_call_results = [
            {
                'role': 'tool',
                'content': str(tool_result),
                'tool_call_id': tool_call['id'],
                'name': tool_call['function']['name'],
            } for tool_call, tool_result in zip(
                input['tool_calls'], input['tool_calls_results']
            )
        ]
        locals['_lazyllm_agent']['workspace']['tool_calls'] = input['tool_calls']
        locals['_lazyllm_agent']['workspace']['tool_call_results'] = tool_call_results
        input = {'input': tool_call_results}
    else:
        locals['_lazyllm_agent']['workspace']['query'] = input
    return input, history, tools, label

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
        self._prompter = ChatPrompter(instruction=_prompt or FC_PROMPT, tools=self._tools_manager.tools_description)\
            .pre_hook(function_call_hook)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter()).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.ins = StreamResponse('Received instruction:', prefix_color=Color.yellow,
                                            color=Color.green, stream=stream)
            self._impl.llm = self._llm
            self._impl.dis = StreamResponse('Decision-making or result in this round:',
                                            prefix_color=Color.yellow, color=Color.green, stream=stream)
            self._impl.post_action = self._post_action

    def _post_action(self, llm_output: Dict[str, Any]):
        if llm_output.get('tool_calls'):
            llm_output['tool_calls_results'] = self._tools_manager(llm_output['tool_calls'])
            locals['_lazyllm_agent']['workspace']['tool_call_trace'].append(
                {
                    'tool_calls': llm_output['tool_calls'],
                    'tool_call_results': llm_output['tool_calls_results'],
                }
            )
        else:
            llm_output = llm_output['content']
        return llm_output

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if isinstance(input, str):
            locals['_lazyllm_agent']['completed'].append(dict(input=input))
            locals['_lazyllm_agent']['workspace'] = dict(tool_call_trace=[])
        globals['chat_history'].setdefault(self._llm._module_id, [])
        if llm_chat_history is not None:
            globals['chat_history'][self._llm._module_id] = llm_chat_history
        result = self._impl(input)
        if isinstance(result, str):
            locals['_lazyllm_agent']['completed'][-1].update(
                {
                    'result': result,
                    'tool_call_trace': locals['_lazyllm_agent']['workspace']['tool_call_trace'],
                }
            )
        return result

class FunctionCallAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        self._fc = FunctionCall(llm, tools, return_trace=return_trace, stream=stream)
        self._agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._fc._llm.used_by(self._module_id)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history) if llm_chat_history is not None else self._agent(query)
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(ValueError(f'After retrying \
            {self._max_retries} times, the function call agent still fails to call successfully.'))
