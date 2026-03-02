import re
from lazyllm.module import ModuleBase
from .base import LazyLLMAgentBase
from lazyllm.components import ChatPrompter
from lazyllm import loop, pipeline, _0, package, bind, LOG, Color, once_wrapper
from .functionCall import FunctionCall, FC_PROMPT
from typing import List, Optional, Union
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
                 workspace: Optional[str] = None, sandbox: Optional[LazyLLMSandboxBase] = None):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls,
                         skills=skills, desc=desc, workspace=workspace,
                         sandbox=sandbox)
        self._assert_tools()
        plan_llm, solve_llm = self._normalize_llms(llm, plan_llm, solve_llm)
        self._init_planner_prompter()
        self._plan_llm = plan_llm.share(prompt=self._planner_prompter, stream=self._planner_stream)\
            .used_by(self._module_id)
        self._solve_llm = solve_llm.share().used_by(self._module_id)
        prompt = FC_PROMPT
        self._fc = FunctionCall(llm=self._solve_llm, return_trace=return_trace, stream=stream,
                                _prompt=prompt, _tool_manager=self._tools_manager,
                                skill_manager=self._skill_manager, workspace=self.workspace)

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
        planner_prompt = self._append_skills_prompt(planner_prompt)
        planner_extra_keys = ['available_skills'] if self._skill_manager else None
        self._planner_prompter = ChatPrompter(instruction=planner_prompt, extra_keys=planner_extra_keys)
        self._planner_stream = dict(prefix='I will give a plan first:\n', prefix_color=Color.blue,
                                    color=Color.green) if self._stream else False

    @once_wrapper(reset_on_pickle=True)
    def build_agent(self):
        with pipeline() as agent:
            agent.plan_input = self._wrap_user_input_with_skills
            agent.plan = self._plan_llm
            agent.parse = (lambda text, query: package([], '', [v for v in re.split('\n\\s*\\d+\\. ', text)[1:]],
                           query)) | bind(query=agent.input)
            with loop(stop_condition=lambda pre, res, steps, query: len(steps) == 0) as agent.lp:
                agent.lp.pre_action = self._pre_action
                agent.lp.solve = loop(self._fc, stop_condition=lambda x: isinstance(x, str),
                                      count=self._max_retries)
                agent.lp.post_action = self._post_action | bind(agent.lp.input[0][0], _0,
                                                                agent.lp.input[0][2],
                                                                agent.lp.input[0][3])

            agent.final_action = lambda pre, res, steps, query: res
        self._agent = agent

    def _pre_action(self, pre_steps, response, steps, query):
        solver_prompt = SOLVER_PROMPT.format(
            previous_steps='\n'.join(pre_steps),
            current_task=steps[0],
            objective=query,
        )
        if self._return_last_tool_calls:
            solver_prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        return package(self._wrap_user_input_with_skills(solver_prompt), [])

    def _build_planner_prompt(self) -> str:
        tools_desc = []
        for name, tool in self._tools_manager.tools_info.items():
            desc = (tool.description or '').strip().splitlines()[0] if tool.description else ''
            tools_desc.append(f'- {name}: {desc}')
        tools_block = '\n'.join(tools_desc)
        if tools_block:
            return f'{PLANNER_PROMPT}\n\nAvailable tools:\n{tools_block}'
        return PLANNER_PROMPT

    def _post_action(self, pre_steps: List[str], response: str, steps: List[str], query: str):
        assert isinstance(response, str), f'After retrying \
            {self._max_retries} times, the solver still failes to call successfully.'
        LOG.debug(f'current step: {steps[0]}, response: {response}')
        current_res = f'- **SubTask**: {steps.pop(0)} **Response**: {response}'
        pre_steps.append(current_res)
        return package(pre_steps, response, steps, query)

    def _post_process(self, result):
        completed = self._pop_tool_calls()
        if completed is not None:
            return completed
        return result
