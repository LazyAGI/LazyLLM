from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import LOG, globals as lazyllm_globals, pipeline, loop, locals, package, FileSystemQueue, once_wrapper
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable, Optional
from .base import LazyLLMAgentBase, _write_agent_data, _unwrap_tool_result
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.common.deprecated import deprecated
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, create_sandbox
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
        if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'\n{self.prefix}\n'}))
        if len(inputs) == 1:
            if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'{inputs[0]}'}))
            return inputs[0]
        if self.stream: FileSystemQueue().enqueue(json.dumps({'tag': 'text', 'delta': f'{inputs}'}))
        return package(*inputs)


_COMPACTION_TRUNCATE_LEN = 200  # chars kept per old tool result


def _tool_result_stop_text(result: Any) -> Optional[str]:
    value = _unwrap_tool_result(result)
    if not isinstance(value, dict):
        return None
    control = value.get('_agent_control')
    if not isinstance(control, dict) or control.get('stop') is not True:
        return None
    final_text = control.get('final_text')
    if not isinstance(final_text, str) or not final_text.strip():
        return None
    return final_text.strip()


def _split_current_tool_input(
    chat_history: List[Dict[str, Any]],
    tool_call_results: List[Dict[str, Any]],
) -> tuple:
    # Compacted history includes current-round tools; ChatPrompter also extends
    # `input` onto history. Split those tools out so they are not duplicated.
    id_order = [item.get('tool_call_id') for item in tool_call_results]
    id_set = {item_id for item_id in id_order if item_id is not None}
    by_id = {}
    remainder = []
    unmatched_tools = []
    for message in chat_history:
        tool_id = message.get('tool_call_id')
        if message.get('role') == 'tool' and tool_id in id_set:
            by_id[tool_id] = message
        elif message.get('role') == 'tool' and not tool_id and id_set:
            unmatched_tools.append(message)
        else:
            remainder.append(message)
    ordered = []
    unused = list(unmatched_tools)
    for original in tool_call_results:
        tool_id = original.get('tool_call_id')
        if tool_id in by_id:
            ordered.append(by_id[tool_id])
        elif unused:
            ordered.append(unused.pop(0))
        else:
            ordered.append(original)
    return remainder, ordered


def _compact_chat_history(history: List[Dict[str, Any]], keep_full_turns: int) -> List[Dict[str, Any]]:
    # identify tool-result message indices (role == 'tool'), from oldest to newest
    tool_indices = [i for i, m in enumerate(history) if m.get('role') == 'tool']
    # keep the last keep_full_turns tool results intact; truncate the rest
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


class FunctionCall(ModuleBase):

    def __init__(self, llm, tools: Optional[List[Union[str, Callable]]] = None, *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None, _tool_manager: Optional[ToolManager] = None,
                 skill_manager=None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 keep_full_turns: int = 0, stop_tools: Optional[List[str]] = None,
                 round_limit: Optional[int] = None,
                 history_compactor: Optional[Callable[[List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
                 runtime_observer: Optional[Callable[..., Any]] = None):
        super().__init__(return_trace=return_trace)
        if _tool_manager is None:
            assert tools, 'tools cannot be empty.'
            self._sandbox = sandbox or create_sandbox()
            self._tools_manager = ToolManager(tools, return_trace=return_trace, sandbox=self._sandbox)
        else:
            self._tools_manager = _tool_manager
            self._sandbox = _tool_manager.sandbox
        self._skill_manager = skill_manager
        self._stream = stream
        self._keep_full_turns = keep_full_turns
        self._history_compactor = history_compactor
        self._stop_tools: set = set(stop_tools) if stop_tools else set()
        self._round_limit = round_limit
        self._runtime_observer = runtime_observer
        prompt = _prompt or FC_PROMPT
        self._prompter = ChatPrompter(
            instruction={'system': prompt, 'user': ''},
            tools=lambda: self._tools_manager.tools_description,
            skills=self._skill_manager.build_prompt() if self._skill_manager else '',
        )
        self._llm = llm.share(
            prompt=self._prompter,
            format=FunctionCallFormatter(),
            stream=stream,
        ).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.pre_action = self._build_history
            self._impl.llm = self._llm
            self._impl.post_action = self._post_action

    @property
    def sandbox(self) -> LazyLLMSandboxBase:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: Optional[LazyLLMSandboxBase]):
        self._sandbox = sandbox
        if hasattr(self, '_tools_manager') and self._tools_manager is not None:
            self._tools_manager.sandbox = sandbox

    def _observe_runtime(self, event: str, **payload):
        if self._runtime_observer is None:
            return
        try:
            self._runtime_observer(event, **payload, sid=lazyllm_globals._sid)
        except Exception:
            pass

    def _prepare_round(self, workspace: Dict[str, Any]) -> tuple:
        if self._round_limit is None:
            return None, None
        current_round = int(workspace.get('_react_round_number', 0)) + 1
        workspace['_react_round_number'] = current_round
        round_limit = int(workspace.get('_react_round_limit', self._round_limit))
        remaining_rounds = max(0, round_limit - current_round)
        LOG.info(
            f'[ReactAgent] [ROUND_BUDGET] sid={lazyllm_globals._sid} current_round={current_round} '
            f'round_limit={round_limit} remaining_rounds={remaining_rounds}'
        )
        self._observe_runtime(
            'turn_start',
            round=current_round,
            round_limit=round_limit,
            remaining_rounds=remaining_rounds,
        )
        return current_round, f'[Internal runtime notice] Internal ReAct rounds left: {remaining_rounds}.'

    def _compact_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._history_compactor is not None:
            return self._history_compactor(history, self._keep_full_turns)
        if self._keep_full_turns > 0:
            return _compact_chat_history(history, self._keep_full_turns)
        return history

    def _notify_history_ready(
        self,
        workspace: Dict[str, Any],
        current_round: Optional[int],
        history: List[Dict[str, Any]],
    ):
        self._observe_runtime(
            'history_ready',
            round=current_round or workspace.get('_react_round_number'),
            history=history,
        )

    def _build_tool_call_input(
        self,
        input: Dict[str, Any],
        workspace: Dict[str, Any],
        budget_notice: Optional[str],
        current_round: Optional[int],
    ) -> Dict[str, Any]:
        tool_call_results = [
            {
                'role': 'tool',
                'content': str(_unwrap_tool_result(tool_call['tool_call_result'])),
                'tool_call_id': tool_call['id'],
                'name': tool_call['function']['name'],
            } for tool_call in workspace['tool_call_trace']
        ]
        if budget_notice and tool_call_results:
            tool_call_results[-1] = {
                **tool_call_results[-1],
                'content': f'{tool_call_results[-1]["content"]}\n\n{budget_notice}',
            }
        workspace['history'].append({
            'role': 'assistant',
            'content': input.get('content', ''),
            'tool_calls': input.get('tool_calls', []),
            'reasoning_content': input.get('reasoning_content', ''),
        })
        workspace['history'].extend(tool_call_results)
        chat_history = self._compact_history(workspace['history'][:])
        remainder, compacted_tools = _split_current_tool_input(chat_history, tool_call_results)
        locals['chat_history'][self._llm._module_id] = remainder
        self._notify_history_ready(workspace, current_round, remainder + compacted_tools)
        return {'input': compacted_tools}

    def _build_history(self, input: Union[str, dict, list]):
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        current_round, budget_notice = self._prepare_round(workspace)

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
            return self._build_tool_call_input(input, workspace, budget_notice, current_round)
        chat_history = self._compact_history(workspace['history'][:history_idx])
        locals['chat_history'][self._llm._module_id] = chat_history
        self._notify_history_ready(workspace, current_round, chat_history)
        return input

    def _post_action(self, llm_output: Dict[str, Any]):  # noqa: C901
        if not llm_output.get('tool_calls'):
            if (match := re.search(r'Action:\s*Call\s+(\w+)\s+with\s+parameters\s+(\{.*?\})', llm_output['content'])):
                try:
                    llm_output['tool_calls'] = [{'function': {'name': match.group(1),
                                                              'arguments': json.loads(match.group(2))}}]
                except Exception: pass
        has_tools = bool(llm_output.get('tool_calls'))
        if tool_calls := llm_output.get('tool_calls'):
            if isinstance(tool_calls, list): [item.pop('index', None) for item in tool_calls]
            tool_calls = self._tools_manager._normalize_tool_calls(tool_calls)
            llm_output['tool_calls'] = tool_calls
            if self._stream:
                _write_agent_data('tool_calls', tool_calls=tool_calls)
            tool_calls_results = self._tools_manager(tool_calls)
            if self._stream:
                _write_agent_data('tool_results',
                                  tool_results=LazyLLMAgentBase._normalize_tool_results(tool_calls,
                                                                                        tool_calls_results))
            locals['_lazyllm_agent']['workspace']['tool_call_trace'] = [
                {**tool_call, 'tool_call_result': tool_result}
                for tool_call, tool_result in zip(tool_calls, tool_calls_results)
            ]
            controlled_stop_texts = [
                text for result in tool_calls_results
                if (text := _tool_result_stop_text(result)) is not None
            ]
            if controlled_stop_texts:
                return '\n'.join(controlled_stop_texts)
            if self._stop_tools:
                called_names = {(tc.get('function') or {}).get('name') for tc in tool_calls if isinstance(tc, dict)}
                if called_names & self._stop_tools:
                    # Only stop the ReAct loop when all stop-tool results succeeded.
                    # If any stop-tool returned ok=False (tool raised an exception), fall through
                    # so the LLM receives the error as a tool observation and can retry.
                    stop_failed = any(
                        isinstance(r, dict) and not r.get('ok', True)
                        for tc, r in zip(tool_calls, tool_calls_results)
                        if isinstance(tc, dict)
                        and (tc.get('function') or {}).get('name') in self._stop_tools
                    )
                    if not stop_failed:
                        if self._runtime_observer:
                            try:
                                workspace = locals['_lazyllm_agent']['workspace']
                                self._runtime_observer(
                                    'turn_end',
                                    round=workspace.get('_react_round_number'),
                                    has_tools=True,
                                    stop_tool=True,
                                    sid=lazyllm_globals._sid,
                                )
                            except Exception:
                                pass
                        return '\n'.join(str(_unwrap_tool_result(r)) for r in tool_calls_results)
        else:
            llm_output = llm_output['content']
        if self._runtime_observer:
            try:
                workspace = locals['_lazyllm_agent']['workspace']
                self._runtime_observer(
                    'turn_end',
                    round=workspace.get('_react_round_number'),
                    has_tools=has_tools,
                    final=not has_tools,
                    sid=lazyllm_globals._sid,
                )
            except Exception:
                pass
        return llm_output

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        workspace = locals['_lazyllm_agent'].setdefault('workspace', {})
        workspace.setdefault('history', list(llm_chat_history or []))
        if self._round_limit is not None and llm_chat_history is not None:
            previous_round = workspace.get('_react_round_number')
            preserved_history_messages = len(workspace.get('history') or [])
            workspace.pop('_react_round_number', None)
            workspace.pop('_react_round_limit', None)
            LOG.info(
                f'[ReactAgent] [INVOCATION_BUDGET_RESET] sid={lazyllm_globals._sid} '
                f'resumed_workspace={previous_round is not None} '
                f'preserved_history_messages={preserved_history_messages} '
                f'new_round_limit={self._round_limit or 0}'
            )
        self._tools_manager.sync_active_groups(input, llm_chat_history)
        try:
            result = self._impl(input)
        except Exception:
            # On failure, clear any in-progress workspace and the LLM chat history so that
            # the next call (e.g. user says "continue") does not inherit a corrupted history
            # that may contain truncated tool_calls with invalid JSON arguments.
            locals['_lazyllm_agent'].pop('workspace', None)
            locals['chat_history'][self._llm._module_id] = []
            raise

        # If the model decides not to call any tools, the result is a string. For debugging and subsequent tasks,
        # the last non-empty tool call trace is stored in locals['_lazyllm_agent']['completed']
        # and history is stored in locals['_lazyllm_agent']['history'].
        if isinstance(result, str):
            workspace = locals['_lazyllm_agent'].pop('workspace', {})
            locals['_lazyllm_agent']['completed'] = workspace.pop(
                'tool_call_trace', locals['_lazyllm_agent'].get('completed', []))
            locals['_lazyllm_agent']['history'] = workspace.pop('history', [])
            locals['chat_history'][self._llm._module_id] = []
        return result

@deprecated('ReactAgent')
class FunctionCallAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None, fs: Optional[Any] = None,
                 skills_dir: Optional[str] = None, enable_builtin_tools: bool = True):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace, fs=fs, skills_dir=skills_dir,
                         enable_builtin_tools=enable_builtin_tools)
        assert self._llm is not None, 'llm cannot be empty.'
        self._assert_tools()
        prompt = self._append_workspace_prompt(FC_PROMPT)
        self._fc = FunctionCall(llm=self._llm, return_trace=return_trace, stream=stream,
                                _prompt=prompt, _tool_manager=self._tools_manager,
                                skill_manager=self._skill_manager)
        self._fc._llm.used_by(self._module_id)

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries + 1)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None:
            return (query, llm_chat_history)
        return query

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the function call agent still fails to call '
                         f'successfully.')
