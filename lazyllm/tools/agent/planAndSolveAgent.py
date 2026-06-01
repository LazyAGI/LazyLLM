import re
from lazyllm.module import ModuleBase
from .base import LazyLLMAgentBase
from .events import PLAN_STARTED, PLAN_FINISHED
from lazyllm.components import ChatPrompter
from lazyllm import LOG, Color, locals
from .functionCall import FC_PROMPT
from typing import List, Any, Optional, Union
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase

PLANNER_PROMPT = (
    'Let\'s first understand the problem and devise a plan to solve it. '
    'Output the plan starting with the header \'Plan:\' followed by a numbered list of steps. '
    'Keep the plan minimal but actionable. '
    'If the task is a question, the final step should usually be '
    '\'Given the above steps taken, respond to the user\'s original question\'. '
    'For each step, indicate whether a tool is needed and which tool. '
    'Do not assume tool outputs; plan to obtain them via tool calls. '
    'At the end, output \'<END_OF_PLAN>\' and nothing else.'
)

SOLVER_PROMPT = '''### Final Objective
{objective}

### Previous Steps
{previous_steps}

### Current Task
{current_task}

Rules:
- Only address the **Current Task**. Do not answer the Final Objective yet.
- Do not infer or complete future steps.
- If information is missing, request it or use tools to obtain it.

Respond with the result for the Current Task only.
'''


class PlanAndSolveAgent(LazyLLMAgentBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[str] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 fs: Optional[Any] = None, skills_dir: Optional[str] = None,
                 enable_builtin_tools: bool = True):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace,
                         sandbox=sandbox, fs=fs, skills_dir=skills_dir,
                         enable_builtin_tools=enable_builtin_tools)
        self._assert_tools()
        plan_llm, solve_llm = self._normalize_llms(llm, plan_llm, solve_llm)
        self._init_planner_prompter()
        self._plan_llm = plan_llm.share(prompt=self._planner_prompter, stream=self._planner_stream)\
            .used_by(self._module_id)
        self._solve_llm = solve_llm.share().used_by(self._module_id)
        self._tool_prompt = self._append_workspace_prompt(FC_PROMPT)

    def _normalize_llms(self, llm, plan_llm, solve_llm):
        assert (llm is None and plan_llm and solve_llm) or (llm and plan_llm is None), (
            'Either specify only llm '
            'without specify plan and solve, or specify only plan and solve without specifying llm, or specify '
            'both llm and solve. Other situations are not allowed.'
        )
        plan_llm = plan_llm or llm
        solve_llm = solve_llm or llm
        return plan_llm, solve_llm

    def _init_planner_prompter(self):
        planner_prompt = self._build_planner_prompt()
        self._planner_prompter = ChatPrompter(
            instruction={'system': planner_prompt, 'user': ''},
            skills=self._skill_manager.build_prompt() if self._skill_manager else '',
        )
        self._planner_stream = dict(prefix='I will give a plan first:\n', prefix_color=Color.blue,
                                    color=Color.green) if self._stream else False

    def _parse_plan_steps(self, plan: str) -> List[str]:
        return [step for step in re.split('\n\\s*\\d+\\. ', plan)[1:] if step]

    def _build_solver_prompt_for_step(self, pre_steps: List[str], step: str, query: str) -> str:
        solver_prompt = SOLVER_PROMPT.format(
            previous_steps='\n'.join(pre_steps),
            current_task=step,
            objective=query,
        )
        if self._return_last_tool_calls:
            solver_prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        return solver_prompt

    def _build_planner_prompt(self) -> str:
        tools_desc = []
        for name, tool in self._tools_manager.tools_info.items():
            desc = (tool.description or '').strip().splitlines()[0] if tool.description else ''
            tools_desc.append(f'- {name}: {desc}')
        tools_block = '\n'.join(tools_desc)
        if tools_block:
            return f'{PLANNER_PROMPT}\n\nAvailable tools:\n{tools_block}'
        return PLANNER_PROMPT

    def _plan_query(self, query: str):
        plan = self._plan_llm(query)
        steps = self._parse_plan_steps(plan)
        return plan, steps

    def _execute(self, input, callback=None):
        locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}
        self._init_tool_llm(prompt=self._tool_prompt, llm=self._solve_llm)

        if callback:
            callback(self._make_event(PLAN_STARTED))
        plan, steps = self._plan_query(input)
        if callback:
            callback(self._make_event(PLAN_FINISHED, text=plan, metadata={'plan': plan, 'steps': steps}))

        pre_steps = []
        response = ''
        for step in steps:
            prompt = self._build_solver_prompt_for_step(pre_steps, step, input)
            for retry_idx in range(self._max_retries):
                history = [] if retry_idx == 0 else None
                response = self._run_tool_round(prompt, history, callback=callback)
                if isinstance(response, str):
                    break
                prompt = response
            else:
                raise AssertionError(f'After retrying {self._max_retries} times, '
                                     f'the solver still failes to call successfully.')
            LOG.debug(f'current step: {step}, response: {response}')
            pre_steps.append(f'- **SubTask**: {step} **Response**: {response}')

        completed = self._pop_tool_calls()
        return completed if completed is not None else response
