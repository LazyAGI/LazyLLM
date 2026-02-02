from lazyllm.module import ModuleBase
from .base import LazyLLMAgentBase
from lazyllm import pipeline, LOG, bind, Color, locals, ifs
from typing import List, Dict, Union, Callable
import re
import json

P_PROMPT_PREFIX = ('For the following tasks, make plans that can solve the problem step-by-step. '
                   'For each plan, indicate which external tool together with tool input to retrieve '
                   'evidence. You can store the evidence into a variable #E that can be called by '
                   'later tools. (Plan, #E1, Plan, #E2, Plan, #E3...) \n\n')

P_FEWSHOT = '''For example,
Task: We are planning to visit the capital city of China this week. What clothing should we wear for the trip?
Plan: First, search for the capital city of China.
#E1 = search[{"query": "What\'s the capital city of China?"}]
Plan: Next, obtain the weather forecast for this week in the capital city of China.
#E2 = weather[{"location": "#E1", "days": 7}]
Plan: Finally, use a language model to generate clothing recommendations based on the weekly weather.
#E3 = llm[{"input": "Using the 7-day forecast in #E2, provide clothing suggestions for visiting #E1 this week."}]'''

P_PROMPT_SUFFIX = '''Begin! Describe your plans with rich details. Each Plan should be followed by only one #E,
and the params_dict is the input of the tool, should be a valid json string wrapped in [],
(e.g. [{{'input': 'hello world', 'num_beams': 5}}]).\n\n'''

S_PROMPT_PREFIX = ('Solve the following task or problem. To assist you, we provide some plans and '
                   'corresponding evidences that might be helpful. Notice that some of these information '
                   'contain noise so you should trust them with caution.\n\n')

S_PROMPT_SUFFIX = ('\nNow begin to solve the task or problem. Respond with '
                   'the answer directly with no extra words.\n\n')

class ReWOOAgent(LazyLLMAgentBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[Union[str, Callable]] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 return_trace: bool = False, stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = ''):
        super().__init__(llm=llm, tools=tools, return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls, skills=skills, desc=desc)
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
        assert self._tools, 'tools cannot be empty.'
        self._planner = plan_llm.share(stream=dict(
            prefix='\nI will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._solver = solve_llm.share(stream=dict(
            prefix='\nI will solve the problem:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._agent = self.build_agent()

    def build_agent(self):
        with pipeline() as agent:
            agent.planner_pre_action = self._build_planner_prompt
            agent.planner = self._planner
            agent.worker_evidences = self._get_worker_evidences
            agent.solver_pre_action = self._build_solver_prompt | bind(input=agent.input)
            agent.solver = ifs(self._return_last_tool_calls, lambda x: 'ok', self._solver)
        return agent

    def _build_planner_prompt(self, input: str):
        prompt = P_PROMPT_PREFIX + 'Tools can be one of the following:\n'
        for name, tool in self._tools_manager.tools_info.items():
            prompt += f'{name}[params_dict]: {tool.description}\n'
        extra_prompt = self._build_extra_system_prompt(input)
        if extra_prompt:
            prompt = f'{prompt}\n\n{extra_prompt}'
        prompt += P_FEWSHOT + '\n' + P_PROMPT_SUFFIX + input + '\n'
        locals['chat_history'][self._planner._module_id] = []
        return prompt

    def _parse_and_call_tool(self, tool_call: str, evidence: Dict[str, str]):
        tool_name, tool_arguments = tool_call.split('[', 1)
        tool_arguments = tool_arguments.split(']')[0]
        for var in re.findall(r'#E\d+', tool_arguments):
            if var in evidence:
                tool_arguments = tool_arguments.replace(var, str(evidence[var]))
        tool_calls = [{'function': {'name': tool_name, 'arguments': tool_arguments}}]
        result = self._tools_manager(tool_calls)
        locals['_lazyllm_agent']['workspace']['tool_call_trace'].append(
            {**tool_calls[0], 'tool_call_result': result[0]}
        )
        return json.dumps(result[0]).strip('\"')

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
        skills_prompt = self._build_extra_system_prompt(input)
        prompt = S_PROMPT_PREFIX + input + '\n' + worker_evidences + S_PROMPT_SUFFIX + input + '\n'
        if skills_prompt:
            prompt = f'{skills_prompt}\n\n{prompt}'
        locals['chat_history'][self._solver._module_id] = []
        return prompt

    def _pre_process(self, query: str):
        locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}
        return query

    def _post_process(self, result):
        trace = self._pop_workspace_tool_calls()
        if trace is not None:
            return trace
        return result
