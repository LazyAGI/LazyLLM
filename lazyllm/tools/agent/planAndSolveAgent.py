import re
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm import loop, pipeline, _0, package, bind, LOG, Color
from .functionCall import FunctionCall
from typing import List, Union, Dict, Any

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

class PlanAndSolveAgent(ModuleBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[str] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 max_retries: int = 5, return_trace: bool = False, stream: bool = False,
                 return_tool_call_results: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert (llm is None and plan_llm and solve_llm) or (llm and plan_llm is None), (
            'Either specify only llm '
            'without specify plan and solve, or specify only plan and solve without specifying llm, or specify '
            'both llm and solve. Other situations are not allowed.'
        )
        assert tools, 'tools cannot be empty.'
        s = dict(prefix='I will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False
        self._plan_llm = ((plan_llm or llm).share(prompt=ChatPrompter(instruction=PLANNER_PROMPT),
                                                  stream=s).used_by(self._module_id))
        self._solve_llm = (solve_llm or llm).share().used_by(self._module_id)
        self._tools = tools
        self._return_tool_call_results = return_tool_call_results
        self._fc = FunctionCall(
            self._solve_llm,
            tools=self._tools,
            return_trace=return_trace,
            stream=stream,
            return_tool_call_results=return_tool_call_results,
        )
        with pipeline() as self._agent:
            self._agent.plan = self._plan_llm
            self._agent.parse = (lambda text, query: package([], '', [v for v in re.split('\n\\s*\\d+\\. ', text)[1:]],
                                 query, [])) | bind(query=self._agent.input)
            with loop(
                stop_condition=lambda pre, res, steps, query, tool_call_results: len(steps) == 0
            ) as self._agent.lp:
                self._agent.lp.pre_action = self._pre_action
                with loop(
                    stop_condition=lambda x, llm_chat_history, tool_call_results: isinstance(x, str),
                    count=self._max_retries,
                ) as self._agent.lp.solve:
                    self._agent.lp.solve.pre_action = (
                        lambda x, llm_chat_history, tool_call_results: package(x, llm_chat_history)
                    )
                    self._agent.lp.solve.act = self._fc
                    self._agent.lp.solve.post_action = self._solve_post_action | bind(
                        llm_chat_history=self._agent.lp.solve.input[0][1],
                        tool_call_results=self._agent.lp.solve.input[0][2],
                    )
                self._agent.lp.post_action = self._post_action | bind(
                    self._agent.lp.input[0][0],
                    _0,
                    self._agent.lp.input[0][2],
                    self._agent.lp.input[0][3],
                    self._agent.lp.input[0][4],
                )

            self._agent.final_action = self._final_action

    def _pre_action(self, pre_steps, response, steps, query, tool_call_results):
        result = SOLVER_PROMPT.format(
            previous_steps='\n'.join(pre_steps),
            current_step=steps[0],
            objective=query,
        )
        return package(result, [], tool_call_results)

    def _post_action(self, pre_steps: List[str], response: str, steps: List[str], query: str, tool_call_results):
        LOG.debug(f'current step: {steps[0]}, response: {response}')
        pre_steps.append(steps.pop(0) + '\nresponse:' + response)
        return package(pre_steps, response, steps, query, tool_call_results)

    def _solve_post_action(self, response: str, llm_chat_history: List[Dict[str, Any]], tool_call_results: List[str]):
        if self._return_tool_call_results and isinstance(response, list) and isinstance(response[-1], list):
            for item in response[-1]:
                if item.get('role', '') == 'tool':
                    tool_call_results.append(f'tool_name: {item.get('name', '')}, '
                                             f'arguments: {item.get('arguments', '')}, '
                                             f'result: {item.get('content', '')}')
        return package(response, llm_chat_history, tool_call_results)

    def _final_action(self, pre_steps: List[str], response: str, steps: List[str], query: str,
                      tool_call_results: List[str]):
        if self._return_tool_call_results:
            response = 'Tool call trace:\n' + '\n'.join(tool_call_results) + '\n\n' + response
        return response

    def forward(self, query: str):
        return self._agent(query)
