from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import LOG, globals as lazyllm_globals, pipeline, loop, locals, package, FileSystemQueue, once_wrapper
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable, Optional
from .base import (
    LazyLLMAgentBase,
    TOOL_OBSERVATION_KEY,
    _model_facing_prefix,
    attachable_tool_observation,
    is_tool_result_envelope,
    strip_tool_observations,
    _write_agent_data,
    _unwrap_tool_result,
)
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.common.deprecated import deprecated
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase, create_sandbox
import re
import json
import inspect

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


_ROUND_TOOLS_KEY = '_function_call_round_tools'
_MODEL_CONTEXT_MAX_CHARS = 2048
_MODEL_CONTEXT_RESERVED_TOKENS = 512


def _structured_compact_parts(compacted: Any) -> Optional[tuple]:
    if not isinstance(compacted, tuple) or len(compacted) != 2:
        return None
    prior_part, current_part = compacted
    if isinstance(prior_part, list) and isinstance(current_part, list):
        return prior_part, current_part
    return None


def _tool_result_observation(result: Any) -> Any:
    if is_tool_result_envelope(result):
        if result['ok']:
            return result.get('value', '')
        if 'value' in result:
            return result.get('value', '')
        return str(result.get('msg', repr(result)))
    return _unwrap_tool_result(result)


def _tool_result_stop_text(result: Any) -> Optional[str]:
    value = _tool_result_observation(result)
    if not isinstance(value, dict):
        return None
    control = value.get('_agent_control')
    if not isinstance(control, dict) or control.get('stop') is not True:
        return None
    final_text = control.get('final_text')
    if not isinstance(final_text, str) or not final_text.strip():
        return None
    return final_text.strip()


class FunctionCall(ModuleBase):

    def __init__(self, llm, tools: Optional[List[Union[str, Callable]]] = None, *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None, _tool_manager: Optional[ToolManager] = None,
                 skill_manager=None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 keep_full_turns: int = 0, stop_tools: Optional[List[str]] = None,
                 round_limit: Optional[int] = None,
                 history_compactor: Optional[Callable[..., Any]] = None,
                 runtime_observer: Optional[Callable[..., Any]] = None,
                 model_context_provider: Optional[Callable[[], Optional[str]]] = None):
        super().__init__(return_trace=return_trace)
        if _tool_manager is None:
            assert tools, 'tools cannot be empty.'
            self._sandbox = sandbox or create_sandbox()
            self._tools_manager = ToolManager(
                tools, return_trace=return_trace, sandbox=self._sandbox,
            )
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
        self._model_context_provider = model_context_provider
        prompt = _prompt or FC_PROMPT
        self._system_prompt = prompt
        self._prompter = ChatPrompter(
            instruction={'system': prompt, 'user': ''},
            tools=self._get_current_tools,
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

    def _get_current_tools(self, refresh: bool = False):
        snapshots = locals.get(_ROUND_TOOLS_KEY)
        if not isinstance(snapshots, dict):
            snapshots = {}
            locals[_ROUND_TOOLS_KEY] = snapshots
        if refresh or self._module_id not in snapshots:
            snapshots[self._module_id] = tuple(self._tools_manager.tools_description)
        return list(snapshots[self._module_id])

    def _get_visible_tool_names(self):
        return {
            item.get('function', {}).get('name')
            for item in self._get_current_tools()
            if item.get('function', {}).get('name')
        }

    def _observe_runtime(self, event: str, **payload):
        if self._runtime_observer is None:
            return
        try:
            self._runtime_observer(event, **payload, sid=lazyllm_globals._sid)
        except Exception:
            pass

    def _prepare_round(self, workspace: Dict[str, Any]) -> tuple:
        if self._round_limit is None:
            return None, None, None
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
        return (
            current_round,
            f'[Internal runtime notice] Internal ReAct rounds left: {remaining_rounds}.',
            remaining_rounds,
        )

    def _compact_history(  # noqa: C901
        self,
        prior_history: List[Dict[str, Any]],
        current_input: Any = None,
        current_round_messages: Optional[List[Dict[str, Any]]] = None,
        workspace: Optional[Dict[str, Any]] = None,
        remaining_rounds: Optional[int] = None,
    ) -> tuple:
        current = list(current_round_messages or [])
        if self._history_compactor is None:
            return strip_tool_observations(prior_history), strip_tool_observations(current)
        prefix = _model_facing_prefix(
            self._system_prompt,
            self._tools_manager,
            self._skill_manager,
        )
        kwargs = {
            'prefix': prefix,
            'current_input': current_input,
            'current_round_messages': current,
        }
        if self._model_context_provider is not None:
            kwargs['reserved_runtime_context_tokens'] = _MODEL_CONTEXT_RESERVED_TOKENS
        try:
            signature = inspect.signature(self._history_compactor)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if workspace is not None and (
                accepts_kwargs or 'runtime_state' in signature.parameters
            ):
                kwargs['runtime_state'] = workspace.setdefault('_history_projection_state', {})
            if remaining_rounds is not None and (
                accepts_kwargs or 'remaining_rounds' in signature.parameters
            ):
                kwargs['remaining_rounds'] = remaining_rounds
            if not accepts_kwargs and 'current_round_messages' not in signature.parameters:
                kwargs.pop('current_round_messages', None)
            if not accepts_kwargs and 'current_input' not in signature.parameters:
                kwargs.pop('current_input', None)
            if not accepts_kwargs and 'prefix' not in signature.parameters:
                kwargs.pop('prefix', None)
            if not accepts_kwargs and 'reserved_runtime_context_tokens' not in signature.parameters:
                kwargs.pop('reserved_runtime_context_tokens', None)
        except (TypeError, ValueError):
            pass
        compacted = self._history_compactor(
            prior_history,
            self._keep_full_turns,
            **kwargs,
        )
        split = _structured_compact_parts(compacted)
        if split is not None:
            return strip_tool_observations(split[0]), strip_tool_observations(split[1])
        compacted = strip_tool_observations(compacted)
        prior_len = len(prior_history)
        if current and len(compacted) == prior_len + len(current):
            return compacted[:prior_len], compacted[prior_len:]
        return compacted, strip_tool_observations(current)

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

    def _consume_model_context(self) -> List[Dict[str, str]]:
        if self._model_context_provider is None:
            return []
        try:
            content = self._model_context_provider()
        except Exception as error:
            LOG.warning(
                f'[ModelContextProvider] failed: {type(error).__name__}: {error}'
            )
            return []
        if not isinstance(content, str) or not content.strip():
            return []
        content = content.strip()
        if len(content) > _MODEL_CONTEXT_MAX_CHARS:
            content = content[:_MODEL_CONTEXT_MAX_CHARS]
            LOG.warning('[ModelContextProvider] context was truncated to fit the input budget')
        self._observe_runtime('runtime_context_delivered', context_count=1)
        return [{'role': 'user', 'content': content}]

    def _build_current_tool_messages(
        self,
        workspace: Dict[str, Any],
        budget_notice: Optional[str],
    ) -> List[Dict[str, Any]]:
        tool_call_results = []
        for tool_call in workspace['tool_call_trace']:
            raw_result = tool_call['tool_call_result']
            tool_message = {
                'role': 'tool',
                'content': str(_tool_result_observation(raw_result)),
                'tool_call_id': tool_call['id'],
                'name': tool_call['function']['name'],
            }
            observation = attachable_tool_observation(raw_result)
            if observation is not None:
                tool_message[TOOL_OBSERVATION_KEY] = observation
            tool_call_results.append(tool_message)
        if budget_notice and tool_call_results:
            tool_call_results[-1] = {
                **tool_call_results[-1],
                'content': f'{tool_call_results[-1]["content"]}\n\n{budget_notice}',
            }
        return tool_call_results

    def _build_tool_call_input(
        self,
        input: Dict[str, Any],
        workspace: Dict[str, Any],
        budget_notice: Optional[str],
        current_round: Optional[int],
        remaining_rounds: Optional[int],
    ) -> Dict[str, Any]:
        current_round_messages = self._build_current_tool_messages(workspace, budget_notice)
        workspace['history'].append({
            'role': 'assistant',
            'content': input.get('content', ''),
            'tool_calls': input.get('tool_calls', []),
            'reasoning_content': input.get('reasoning_content', ''),
        })
        prior_history = workspace['history'][:]
        workspace['history'].extend(current_round_messages)
        compacted_prior, compacted_current = self._compact_history(
            prior_history,
            current_input='',
            current_round_messages=current_round_messages,
            workspace=workspace,
            remaining_rounds=remaining_rounds,
        )
        compacted_current.extend(self._consume_model_context())
        locals['chat_history'][self._llm._module_id] = compacted_prior
        self._notify_history_ready(workspace, current_round, compacted_prior + compacted_current)
        return {'input': compacted_current}

    def _build_history(self, input: Union[str, dict, list]):
        self._get_current_tools(refresh=True)
        workspace = locals['_lazyllm_agent']['workspace']
        history_idx = len(workspace.setdefault('history', []))
        current_round, budget_notice, remaining_rounds = self._prepare_round(workspace)

        current_input = None
        if isinstance(input, str):
            workspace['history'].append({'role': 'user', 'content': input})
            current_input = input
        elif isinstance(input, dict) and 'input' in input:
            workspace['history'].append(
                {'role': 'user', 'content': input.get('input', '')}
            )
            current_input = input.get('input', '')
        elif isinstance(input, dict) and input.get('role') == 'user':
            workspace['history'].append(
                {'role': 'user', 'content': input.get('content', '')}
            )
            current_input = input.get('content', '')
        elif isinstance(input, dict):
            return self._build_tool_call_input(
                input,
                workspace,
                budget_notice,
                current_round,
                remaining_rounds,
            )
        compacted_prior, _compacted_current = self._compact_history(
            workspace['history'][:history_idx],
            current_input=current_input,
            workspace=workspace,
            remaining_rounds=remaining_rounds,
        )
        locals['chat_history'][self._llm._module_id] = compacted_prior
        self._notify_history_ready(workspace, current_round, compacted_prior)
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
            tool_calls = self._tools_manager.normalize_tool_calls(tool_calls)
            llm_output['tool_calls'] = tool_calls
            if self._stream:
                _write_agent_data('tool_calls', tool_calls=tool_calls)
            tool_calls_results = self._tools_manager(
                tool_calls,
                allowed_tool_names=self._get_visible_tool_names(),
            )
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
                        return '\n'.join(str(_tool_result_observation(r)) for r in tool_calls_results)
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
