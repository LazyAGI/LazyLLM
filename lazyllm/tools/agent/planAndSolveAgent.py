import re
from lazyllm.module import ModuleBase
from .base import LazyLLMAgentBase
from lazyllm.components import ChatPrompter
from lazyllm import loop, pipeline, _0, package, bind, LOG, Color
from .functionCall import FunctionCall
from typing import List, Union

PLANNER_PROMPT = (
    'Let\'s first understand the problem and devise a plan to solve the problem.'
    ' Please output the plan starting with the header \'Plan:\' '
    'and then followed by a numbered list of steps. '
    'Please make the plan the minimum number of steps required '
    'to accurately complete the task. If the task is a question, '
    'the final step should almost always be \'Given the above steps taken, '
    'please respond to the users original question\'. '
    'At the end of your plan, say \'<END_OF_PLAN>\'. Just output the plan itself '
    'without any additional prompt information, such as mentioning that tools cannot be used directly. '
    'You should not echo any other words after \'<END_OF_PLAN>\''
)

SOLVER_PROMPT = (
    'Objective: {objective}\n\n'
    'Previos steps: {previous_steps}\n\n'
    'Current objective: {current_step}\n\n'
    'Just solve the current objective, don\'t overdo it.'
)

class PlanAndSolveAgent(LazyLLMAgentBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[str] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_last_tool_calls: bool = False):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls)
        if llm is None and plan_llm is None and solve_llm is None:
            raise ValueError('Either specify llm, or provide plan_llm/solve_llm.')
        if llm is None:
            if plan_llm is None:
                plan_llm = solve_llm
            if solve_llm is None:
                solve_llm = plan_llm
        else:
            if plan_llm is None:
                plan_llm = llm
            if solve_llm is None:
                solve_llm = llm
        assert tools, 'tools cannot be empty.'
        s = dict(prefix='I will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False
        self._plan_llm = (plan_llm.share(prompt=ChatPrompter(instruction=PLANNER_PROMPT),
                                         stream=s).used_by(self._module_id))
        self._solve_llm = solve_llm.share().used_by(self._module_id)
        self._fc = FunctionCall(self._solve_llm, tools=self._tools, return_trace=return_trace, stream=stream)
        self._agent = self.build_agent()

    def build_agent(self):
        with pipeline() as agent:
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
        return agent

    def _pre_action(self, pre_steps, response, steps, query):
        solver_prompt = SOLVER_PROMPT.format(
            previous_steps='\n'.join(pre_steps),
            current_step=steps[0],
            objective=query,
        ) + 'input: ' + steps[0]
        if self._return_last_tool_calls:
            solver_prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        return package(solver_prompt, [])

    def _post_action(self, pre_steps: List[str], response: str, steps: List[str], query: str):
        assert isinstance(response, str), f'After retrying \
            {self._max_retries} times, the solver still failes to call successfully.'
        LOG.debug(f'current step: {steps[0]}, response: {response}')
        pre_steps.append(steps.pop(0) + 'response: ' + response)
        return package(pre_steps, response, steps, query)

    def _post_process(self, result):
        completed = self._pop_completed_tool_calls()
        if completed is not None:
            return completed
        return result
