from .base import LazyLLMAgentBase
from lazyllm import loop, once_wrapper, LOG, locals
from .functionCall import FunctionCall
from typing import List, Any, Dict, Optional, Union
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase

INSTRUCTION = f'''
## Role
You are a **ReAct-style autonomous agent** designed to solve user tasks through an iterative closed loop:
Reason → Act → Observe → Reflect.

Your goal is to produce **accurate, efficient, and well-structured results** while making optimal use of available tools.

## Core Working Loop (ReAct)
You must repeatedly follow this loop until the task is fully solved:
### 1. Reason
- Understand the user's intent and constraints.
- Break complex tasks into manageable steps.
- Decide whether external tools are required.
- Plan the next best action.

### 2. Act
- Select **the most appropriate tool** if a tool is needed.
- Invoke a tool **only when it provides capabilities or information you do not already have**.
- Avoid speculative, redundant, or exploratory tool calls.

### 3. Observe
- Carefully examine the tool's output.
- Extract only information relevant to the current task.
- Ignore noise or irrelevant details.

### 4. Reflect
- Evaluate whether the obtained information is sufficient.
- If insufficient, refine the plan and continue the loop.
- If sufficient, prepare the final answer.

## Tool Usage Guidelines
- You have access to multiple tools.
- Before calling a tool, always consider:
  * *Is a tool strictly necessary at this step?*
  * *Which available tool is the most suitable for this purpose?*
- Use **at most one tool per action step**.
- Do **not** call any tools after you already have enough information to answer.

{FC_PROMPT_PLACEHOLDER}

## Language & Communication Rules
- Always respond in **the same language as the user's question**.
- Do not switch languages unless explicitly requested.
- Be concise, precise, and task-focused.
- Do **not** expose internal reasoning, chain-of-thought, or decision deliberations to the user.

## Final Answer Rule
When the task is complete:
  * Stop the ReAct loop.
  * Do not call any additional tools.
  * Provide a clear and complete final answer.
The final response should contain **only the answer itself**, without internal process details.

You are responsible for maintaining correctness, efficiency, and clarity throughout the entire reasoning–action cycle.

'''


_FORCE_SUMMARIZE_MSG = (
    'You have reached the maximum number of tool calls. '
    'Stop calling tools now and provide your final answer immediately.'
)


class ReactAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools: Optional[List[str]] = None, max_retries: int = 5, return_trace: bool = False,
                 prompt: str = None, stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Optional[Union[bool, str, List[str]]] = None, desc: str = '',
                 workspace: Optional[str] = None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 force_summarize: bool = False, force_summarize_context: str = '',
                 keep_full_turns: int = 0):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries, return_trace=return_trace,
                         stream=stream, return_last_tool_calls=return_last_tool_calls, skills=skills,
                         desc=desc, workspace=workspace, sandbox=sandbox)
        prompt = prompt or INSTRUCTION
        if self._return_last_tool_calls:
            prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        assert self._llm is not None, 'llm cannot be empty.'
        self._assert_tools()
        self._prompt = prompt
        self._force_summarize = force_summarize
        self._force_summarize_context = force_summarize_context
        self._keep_full_turns = keep_full_turns

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        agent = loop(FunctionCall(llm=self._llm, _prompt=self._prompt, return_trace=self._return_trace,
                                  stream=self._stream, _tool_manager=self._tools_manager,
                                  skill_manager=self._skill_manager, workspace=self.workspace,
                                  keep_full_turns=self._keep_full_turns),
                     stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._agent = agent

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        return (self._wrap_user_input_with_skills(query), llm_chat_history or [])

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_tool_calls()
            if completed is not None:
                return completed
            return ret
        if self._force_summarize:
            history = locals['_lazyllm_agent'].get('workspace', {}).get('history', [])
            if history:
                LOG.warning(
                    f'ReactAgent reached max_retries={self._max_retries}, attempting force summarize.'
                )
                summary = None
                try:
                    recent = history[-8:]
                    obs_lines: List[str] = []
                    for idx, m in enumerate(recent):
                        role = m.get('role', '')
                        raw_content = m.get('content')
                        if raw_content is None:
                            tool_calls = m.get('tool_calls') or []
                            if tool_calls:
                                names = ', '.join(
                                    tc.get('function', {}).get('name', '?') for tc in tool_calls
                                )
                                obs_lines.append(f'{role}: [called tools: {names}]')
                            continue
                        content_str = raw_content if isinstance(raw_content, str) else str(raw_content)
                        limit = 1000 if idx == len(recent) - 1 else 300
                        obs_lines.append(f'{role}: {content_str[:limit]}')
                    obs_text = '\n'.join(obs_lines)
                    ctx_prefix = (
                        f'Original task context:\n{self._force_summarize_context[:500]}\n\n'
                        if self._force_summarize_context else ''
                    )
                    summarize_prompt = (
                        f'{ctx_prefix}'
                        f'Based on the following recent observations from your tool exploration:\n'
                        f'{obs_text}\n\n'
                        f'{_FORCE_SUMMARIZE_MSG}'
                    )
                    summarize_llm = self._llm.share(stream=False)
                    resp = summarize_llm(summarize_prompt)
                    summary = resp if isinstance(resp, str) else (
                        resp.get('content', '') if isinstance(resp, dict) else None
                    )
                except Exception as e:
                    LOG.warning(f'ReactAgent force-summarize call failed: {e}')
                if summary is not None:
                    return summary
        raise ValueError(f'After retrying {self._max_retries} times, the react agent still failes to call '
                         f'successfully.')
