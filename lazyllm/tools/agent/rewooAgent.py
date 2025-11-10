from lazyllm.module import ModuleBase
from lazyllm import pipeline, LOG, globals, bind, Color, locals
from .toolsManager import ToolManager
from typing import List, Dict, Union, Callable
import re

P_PROMPT_PREFIX = ('For the following tasks, make plans that can solve the problem step-by-step. '
                   'For each plan, indicate which external tool together with tool input to retrieve '
                   'evidence. You can store the evidence into a variable #E that can be called by '
                   'later tools. (Plan, #E1, Plan, #E2, Plan, #E3...) \n\n')

P_FEWSHOT = '''For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha.
#E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked.
#E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked.
#E3 = Calculator[(2 ∗ #E2 − 10) − 8]'''

P_PROMPT_SUFFIX = ('Begin! Describe your plans with rich details. Each Plan should be followed by only one #E, '
                   'and the params_dict is the input of the tool, should be a valid json string wrapped in [].\n\n')

S_PROMPT_PREFIX = ('Solve the following task or problem. To assist you, we provide some plans and '
                   'corresponding evidences that might be helpful. Notice that some of these information '
                   'contain noise so you should trust them with caution.\n\n')

S_PROMPT_SUFFIX = ('\nNow begin to solve the task or problem. Respond with '
                   'the answer directly with no extra words.\n\n')

class ReWOOAgent(ModuleBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[Union[str, Callable]] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 return_trace: bool = False, stream: bool = False, return_tool_call_results: bool = False):
        super().__init__(return_trace=return_trace)
        assert (llm is None and plan_llm and solve_llm) or (llm and plan_llm is None), 'Either specify only llm \
               without specify plan and solve, or specify only plan and solve without specifying llm, or specify \
               both llm and solve. Other situations are not allowed.'
        assert tools, 'tools cannot be empty.'
        self._planner = (plan_llm or llm).share(stream=dict(
            prefix='\nI will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._solver = (solve_llm or llm).share(stream=dict(
            prefix='\nI will solve the problem:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._tools_manager = ToolManager(tools, return_trace=return_trace)
        with pipeline() as self._agent:
            self._agent.planner_pre_action = self._build_planner_prompt
            self._agent.planner = self._planner
            self._agent.worker_evidences = self._get_worker_evidences
            self._agent.solver_pre_action = self._build_solver_prompt | bind(input=self._agent.input)
            self._agent.solver = self._solver

    def _build_planner_prompt(self, input: str):
        prompt = P_PROMPT_PREFIX + 'Tools can be one of the following:\n'
        for name, tool in self._tools_manager.tools_info.items():
            prompt += f'{name}[params_dict]: {tool.description}\n'
        prompt += P_FEWSHOT + '\n' + P_PROMPT_SUFFIX + input + '\n'
        globals['chat_history'][self._planner._module_id] = []
        return prompt

    def _parse_and_call_tool(self, tool_call: str, evidence: Dict[str, str]):
        tool_name, tool_arguments = tool_call.split('[', 1)
        tool_arguments = tool_arguments.split(']')[0]
        for var in re.findall(r'#E\d+', tool_arguments):
            if var in evidence:
                tool_arguments = tool_arguments.replace(var, str(evidence[var]))
        tool_calls = [{'function': {'name': tool_name, 'arguments': tool_arguments}}]
        result = self._tools_manager(tool_calls)
        tool_call_results = [{'role': 'tool', 'content': str(result[0]), 'name': tool_name}]
        locals['_lazyllm_agent']['workspace']['tool_call_trace'].append(
            {
                'tool_calls': tool_calls,
                'tool_call_results': tool_call_results,
            }
        )
        return result[0]

    def _get_worker_evidences(self, response: str):
        LOG.debug(f'planner plans: {response}')
        evidence = {}
        worker_evidences = ''
        for line in response.splitlines():
            if line.startswith('Plan'):
                worker_evidences += line + '\n'
            elif re.match(r'#E\d+\s*=', line.strip()):
                e, tool_call = line.split('=', 1)
                evidence[e.strip()] = self._parse_and_call_tool(tool_call.strip(), evidence)
                worker_evidences += f'Evidence:\n{evidence[e.strip()]}\n'

        LOG.debug(f'worker_evidences: {worker_evidences}')
        return worker_evidences

    def _build_solver_prompt(self, worker_evidences, input):
        prompt = S_PROMPT_PREFIX + input + '\n' + worker_evidences + S_PROMPT_SUFFIX + input + '\n'
        globals['chat_history'][self._solver._module_id] = []
        return prompt

    def forward(self, query: str):
        locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}
        result = self._agent(query)
        locals['_lazyllm_agent']['completed'].append(
            dict(input=query, result=result, tool_call_trace=locals['_lazyllm_agent']['workspace']['tool_call_trace'])
        )
        return result
