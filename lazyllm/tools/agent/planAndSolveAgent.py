import re
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm import loop, pipeline, _0, package, bind, LOG
from .functionCall import FunctionCallAgent
from typing import List

PLANNER_PROMPT = (
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
)

SOLVER_PROMPT = (
    "Objective: {objective}\n\n"
    "Previos steps: {previous_steps}\n\n"
    "Current objective: {current_step}"
)

class PlanAndSolveAgent(ModuleBase):
    def __init__(self, plan_llm, solve_llm, tools: List[str], max_retries: int = 5, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert plan_llm and solve_llm and tools, "llm and tools cannot be empty."
        self._plan_llm = plan_llm
        self._solve_llm = solve_llm
        self._tools = tools
        self._agent = self._build_pipeline()

    def _post_action(self, pre_steps: List[str], response: str, steps: List[str], query: str):
        LOG.info(f"current step: {steps[0]}, response: {response}")
        pre_steps.append(steps.pop(0))
        return package(pre_steps, response, steps, query)

    def _build_pipeline(self):
        with pipeline() as ppl:
            ppl.plan = self._plan_llm.share(prompt=ChatPrompter(instruction=PLANNER_PROMPT))
            ppl.parse = (lambda text, query: package([], '', [v for v in re.split("\n\\s*\\d+\\. ", text)[1:]],
                         query)) | bind(query=ppl.input)
            with loop(stop_condition=lambda pre, res, steps, query: len(steps) == 0) as ppl.lp:
                ppl.lp.pre_action = lambda pre_steps, response, steps, query: \
                    package(SOLVER_PROMPT.format(previous_steps="\n".join(pre_steps), current_step=steps[0],
                            objective=query) + "input: " + response + "\n" + steps[0], [])
                ppl.lp.solve = FunctionCallAgent(self._solve_llm, tools=self._tools)
                ppl.lp.post_action = self._post_action | bind(ppl.lp.input[0][0], _0, ppl.lp.input[0][2],
                                                              ppl.lp.input[0][3])

            ppl.post_action = lambda pre, res, steps, query: res

        return ppl

    def forward(self, query: str):
        ans = self._agent(query)
        return ans
