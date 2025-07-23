import re
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm import loop, pipeline, _0, package, bind, LOG, Color
from .functionCall import FunctionCallAgent
from typing import List, Union

PLANNER_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'. Just output the plan itself "
    "without any additional prompt information, such as mentioning that tools cannot be used directly. "
    "You should not echo any other words after '<END_OF_PLAN>'"
)

SOLVER_PROMPT = (
    "Objective: {objective}\n\n"
    "Previos steps: {previous_steps}\n\n"
    "Current objective: {current_step}\n\n"
    "Just solve the current objective, don't overdo it."
)

class PlanAndSolveAgent(ModuleBase):
    """PlanAndSolveAgent consists of two components. First, the planner breaks down the entire task into smaller subtasks, then the solver executes these subtasks according to the plan, which may involve tool calls, and finally returns the answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and sovler share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, PlanAndSolveAgent
    >>> @fc_register("tool")
    >>> def multiply(a: int, b: int) -> int:
    ...     '''
    ...     Multiply two integers and return the result integer
    ...
    ...     Args:
    ...         a (int): multiplier
    ...         b (int): multiplier
    ...     '''
    ...     return a * b
    ...
    >>> @fc_register("tool")
    >>> def add(a: int, b: int):
    ...     '''
    ...     Add two integers and returns the result integer
    ...
    ...     Args:
    ...         a (int): addend
    ...         b (int): addend
    ...     '''
    ...     return a + b
    ...
    >>> tools = ["multiply", "add"]
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = PlanAndSolveAgent(llm, tools)
    >>> query = "What is 20+(2*4)? Calculate step by step."
    >>> res = agent(query)
    >>> print(res)
    'The final answer is 28.'
    """
    def __init__(self, llm: Union[ModuleBase, None] = None, tools: List[str] = [], *,
                 plan_llm: Union[ModuleBase, None] = None, solve_llm: Union[ModuleBase, None] = None,
                 max_retries: int = 5, return_trace: bool = False, stream: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert (llm is None and plan_llm and solve_llm) or (llm and plan_llm is None), 'Either specify only llm \
               without specify plan and solve, or specify only plan and solve without specifying llm, or specify \
               both llm and solve. Other situations are not allowed.'
        assert tools, "tools cannot be empty."
        s = dict(prefix='I will give a plan first:\n', prefix_color=Color.blue, color=Color.green) if stream else False
        self._plan_llm = ((plan_llm or llm).share(prompt=ChatPrompter(instruction=PLANNER_PROMPT),
                                                  stream=s).used_by(self._module_id))
        self._solve_llm = (solve_llm or llm).share().used_by(self._module_id)
        self._tools = tools
        with pipeline() as self._agent:
            self._agent.plan = self._plan_llm
            self._agent.parse = (lambda text, query: package([], '', [v for v in re.split("\n\\s*\\d+\\. ", text)[1:]],
                                 query)) | bind(query=self._agent.input)
            with loop(stop_condition=lambda pre, res, steps, query: len(steps) == 0) as self._agent.lp:
                self._agent.lp.pre_action = self._pre_action
                self._agent.lp.solve = FunctionCallAgent(self._solve_llm, tools=self._tools,
                                                         return_trace=return_trace, stream=stream)
                self._agent.lp.post_action = self._post_action | bind(self._agent.lp.input[0][0], _0,
                                                                      self._agent.lp.input[0][2],
                                                                      self._agent.lp.input[0][3])

            self._agent.post_action = lambda pre, res, steps, query: res

    def _pre_action(self, pre_steps, response, steps, query):
        result = package(SOLVER_PROMPT.format(previous_steps="\n".join(pre_steps), current_step=steps[0],
                                              objective=query) + "input: " + response + "\n" + steps[0], [])
        return result

    def _post_action(self, pre_steps: List[str], response: str, steps: List[str], query: str):
        LOG.debug(f"current step: {steps[0]}, response: {response}")
        pre_steps.append(steps.pop(0))
        return package(pre_steps, response, steps, query)

    def forward(self, query: str):
        return self._agent(query)
