from lazyllm.module import ModuleBase
from lazyllm import pipeline, package, LOG, globals, bind, Color
from .toolsManager import ToolManager
from typing import List, Dict, Union, Callable
import re

P_PROMPT_PREFIX = ("For the following tasks, make plans that can solve the problem step-by-step. "
                   "For each plan, indicate which external tool together with tool input to retrieve "
                   "evidence. You can store the evidence into a variable #E that can be called by "
                   "later tools. (Plan, #E1, Plan, #E2, Plan, #E3...) \n\n")

P_FEWSHOT = """For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha.
#E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked.
#E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked.
#E3 = Calculator[(2 ∗ #E2 − 10) − 8]"""

P_PROMPT_SUFFIX = ("Begin! Describe your plans with rich details. Each Plan should be followed by only one #E.\n\n")

S_PROMPT_PREFIX = ("Solve the following task or problem. To assist you, we provide some plans and "
                   "corresponding evidences that might be helpful. Notice that some of these information "
                   "contain noise so you should trust them with caution.\n\n")

S_PROMPT_SUFFIX = ("\nNow begin to solve the task or problem. Respond with "
                   "the answer directly with no extra words.\n\n")

class ReWOOAgent(ModuleBase):
    """ReWOOAgent consists of three parts: Planer, Worker and Solver. The Planner uses predictive reasoning capabilities to create a solution blueprint for a complex task; the Worker interacts with the environment through tool calls and fills in actual evidence or observations into instructions; the Solver processes all plans and evidence to develop a solution to the original task or problem.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.


Examples:
    >>> import lazyllm
    >>> import wikipedia
    >>> from lazyllm.tools import fc_register, ReWOOAgent
    >>> @fc_register("tool")
    >>> def WikipediaWorker(input: str):
    ...     '''
    ...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
    ...
    ...     Args:
    ...         input (str): search query.
    ...     '''
    ...     try:
    ...         evidence = wikipedia.page(input).content
    ...         evidence = evidence.split("\\n\\n")[0]
    ...     except wikipedia.PageError:
    ...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    ...     except wikipedia.DisambiguationError:
    ...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    ...     return evidence
    ...
    >>> @fc_register("tool")
    >>> def LLMWorker(input: str):
    ...     '''
    ...     A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
    ...
    ...     Args:
    ...         input (str): instruction
    ...     '''
    ...     llm = lazyllm.OnlineChatModule(source="glm")
    ...     query = f"Respond in short directly with no extra words.\\n\\n{input}"
    ...     response = llm(query, llm_chat_history=[])
    ...     return response
    ...
    >>> tools = ["WikipediaWorker", "LLMWorker"]
    >>> llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(lazyllm.deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = ReWOOAgent(llm, tools)
    >>> query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
    >>> res = agent(query)
    >>> print(res)
    '
    Hennessy '
    """
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[Union[str, Callable]] = [], *,
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 return_trace: bool = False, stream: bool = False):
        super().__init__(return_trace=return_trace)
        assert (llm is None and plan_llm and solve_llm) or (llm and plan_llm is None), 'Either specify only llm \
               without specify plan and solve, or specify only plan and solve without specifying llm, or specify \
               both llm and solve. Other situations are not allowed.'
        assert tools, "tools cannot be empty."
        self._planner = (plan_llm or llm).share(stream=dict(
            prefix='\nI will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._solver = (solve_llm or llm).share(stream=dict(
            prefix='\nI will solve the problem:\n', prefix_color=Color.blue, color=Color.green) if stream else False)
        self._name2tool = ToolManager(tools, return_trace=return_trace).tools_info
        with pipeline() as self._agent:
            self._agent.planner_pre_action = self._build_planner_prompt
            self._agent.planner = self._planner
            self._agent.parse_plan = self._parse_plan
            self._agent.woker = self._get_worker_evidences
            self._agent.solver_pre_action = self._build_solver_prompt | bind(input=self._agent.input)
            self._agent.solver = self._solver

    def _build_planner_prompt(self, input: str):
        prompt = P_PROMPT_PREFIX + "Tools can be one of the following:\n"
        for name, tool in self._name2tool.items():
            prompt += f"{name}[search query]: {tool.description}\n"
        prompt += P_FEWSHOT + "\n" + P_PROMPT_SUFFIX + input + "\n"
        globals['chat_history'][self._planner._module_id] = []
        return prompt

    def _parse_plan(self, response: str):
        LOG.debug(f"planner plans: {response}")
        plans = []
        evidence = {}
        for line in response.splitlines():
            if line.startswith("Plan"):
                plans.append(line)
            elif line.startswith("#") and line[1] == "E" and line[2].isdigit():
                e, tool_call = line.split("=", 1)
                e, tool_call = e.strip(), tool_call.strip()
                if len(e) == 3:
                    evidence[e] = tool_call
                else:
                    evidence[e] = "No evidence found"
        return package(plans, evidence)

    def _get_worker_evidences(self, plans: List[str], evidence: Dict[str, str]):
        worker_evidences = {}
        for e, tool_call in evidence.items():
            if "[" not in tool_call:
                worker_evidences[e] = tool_call
                continue
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1].strip("'").strip('"')
            # find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in worker_evidences:
                    tool_input = tool_input.replace(var, "[" + worker_evidences[var] + "]")
            tool_instance = self._name2tool.get(tool)
            if tool_instance:
                worker_evidences[e] = tool_instance(tool_input)
            else:
                worker_evidences[e] = "No evidence found"

        worker_log = ""
        for idx, plan in enumerate(plans):
            e = f"#E{idx+1}"
            worker_log += f"{plan}\nEvidence:\n{worker_evidences[e]}\n"
        LOG.debug(f"worker_log: {worker_log}")
        return worker_log

    def _build_solver_prompt(self, worker_log, input):
        prompt = S_PROMPT_PREFIX + input + "\n" + worker_log + S_PROMPT_SUFFIX + input + "\n"
        globals['chat_history'][self._solver._module_id] = []
        return prompt

    def forward(self, query: str):
        return self._agent(query)
