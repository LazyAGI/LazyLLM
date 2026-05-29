from typing import Callable, Dict, List, Any, Optional, Union
import re
import json

from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm import LOG, bind, Color, locals, ifs
from lazyllm.tools.sandbox.sandbox_base import LazyLLMSandboxBase
from .base import LazyLLMAgentBase
from .events import PLAN_STARTED, PLAN_FINISHED, TOOL_CALLS, TOOL_RESULTS


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
S_PROMPT_TEMPLATE = S_PROMPT_PREFIX + '{objective}\n{worker_evidences}' + S_PROMPT_SUFFIX + '{objective}\n'

class ReWOOAgent(LazyLLMAgentBase):
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[Union[str, Callable]] = [], *,  # noqa B006
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 return_trace: bool = False, stream: bool = False, return_last_tool_calls: bool = False,
                 skills: Union[bool, str, List[str], None] = None, desc: str = '',
                 workspace: Optional[str] = None, sandbox: Optional[LazyLLMSandboxBase] = None,
                 fs: Optional[Any] = None, skills_dir: Optional[str] = None,
                 enable_builtin_tools: bool = True):
        super().__init__(llm=llm, tools=tools, return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls, skills=skills, desc=desc,
                         workspace=workspace, sandbox=sandbox, fs=fs, skills_dir=skills_dir,
                         enable_builtin_tools=enable_builtin_tools)
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
        self._assert_tools()
        skills_prompt = self._skill_manager.build_prompt() if self._skill_manager else ''
        planner_prompt = self._build_planner_prompt_template()
        self._tool_solver_prompt = self._append_workspace_prompt(S_PROMPT_TEMPLATE)
        self._planner = plan_llm.share(
            prompt=ChatPrompter(instruction={'system': planner_prompt, 'user': ''}, skills=skills_prompt),
            stream=dict(prefix='\nI will give a plan first:\n', prefix_color=Color.blue, color=Color.green)
            if stream else False
        )
        self._solver = solve_llm.share(
            prompt=ChatPrompter(instruction={'system': self._tool_solver_prompt, 'user': ''},
                                skills=skills_prompt),
            stream=dict(prefix='\nI will solve the problem:\n', prefix_color=Color.blue, color=Color.green)
            if stream else False
        )

    def _build_planner_prompt_template(self):
        prompt = P_PROMPT_PREFIX + 'Tools can be one of the following:\n'
        for name, tool in self._tools_manager.tools_info.items():
            prompt += f'{name}[params_dict]: {tool.description}\n'
        prompt += P_FEWSHOT + '\n' + P_PROMPT_SUFFIX
        return prompt

    def _parse_and_call_tool(self, tool_call: str, evidence: Dict[str, str], callback=None):
        tool_name, tool_arguments = tool_call.split('[', 1)
        tool_arguments = tool_arguments.split(']')[0]
        for var in re.findall(r'#E\d+', tool_arguments):
            if var in evidence:
                tool_arguments = tool_arguments.replace(var, str(evidence[var]))
        tool_calls = [{'function': {'name': tool_name, 'arguments': tool_arguments}}]
        if callback:
            callback(self._make_event(TOOL_CALLS, tool_calls=tool_calls))
        result = self._tools_manager(tool_calls)
        if callback:
            callback(self._make_event(TOOL_RESULTS,
                                      tool_results=self._normalize_tool_results(tool_calls, result)))
        locals['_lazyllm_agent']['workspace']['tool_call_trace'].append(
            {**tool_calls[0], 'tool_call_result': result[0]}
        )
        return json.dumps(result[0]).strip('\"')

    def _get_worker_evidences(self, response: str, callback=None):
        LOG.debug(f'planner plans: {response}')
        evidence = {}
        worker_evidences = ''
        for line in response.splitlines():
            if line.startswith('Plan'):
                worker_evidences += line + '\n'
            elif re.match(r'#E\d+\s*=', line.strip()):
                e, tool_call = line.split('=', 1)
                evidence[e.strip()] = self._parse_and_call_tool(tool_call.strip(), evidence, callback=callback)
                worker_evidences += f'Evidence:\n{evidence[e.strip()]}\n'

        LOG.debug(f'worker_evidences: {worker_evidences}')
        return worker_evidences

    def _execute(self, input, callback=None):
        locals['_lazyllm_agent']['workspace'] = {'tool_call_trace': []}

        if callback:
            callback(self._make_event(PLAN_STARTED))
        locals['chat_history'][self._planner._module_id] = []
        plan = self._planner(input)
        if callback:
            callback(self._make_event(PLAN_FINISHED, text=plan, metadata={'plan': plan}))

        worker_evidences = self._get_worker_evidences(plan, callback=callback)
        locals['chat_history'][self._solver._module_id] = []
        solver_input = {'input': input, 'objective': input, 'worker_evidences': worker_evidences}
        result = 'ok' if self._return_last_tool_calls else self._solver(solver_input)

        trace = self._pop_tool_calls()
        return trace if trace is not None else result
