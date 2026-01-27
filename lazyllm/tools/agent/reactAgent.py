from .base import LazyLLMAgentBase
from lazyllm import loop
from .functionCall import FunctionCall
from typing import List, Any, Dict
from lazyllm.components.prompter.builtinPrompt import FC_PROMPT_PLACEHOLDER

INSTRUCTION = f'''You are designed to help with a variety of tasks, from answering questions to providing \
summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence \
you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:

## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help answer the question.
{FC_PROMPT_PLACEHOLDER}
Answering questions should include Thought regardless of whether or not you need to \
call a tool.(Thought is required, tool_calls is optional.)

Please ALWAYS start with a Thought and Only ONE Thought at a time.

You should keep repeating the above format till you have enough information to answer the question without using \
any more tools. At that point, you MUST respond in the following formats:

Answer: your answer here (In the same language as the user's question)

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages. Think step by step.'''


class ReactAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False,
                 prompt: str = None, stream: bool = False, return_last_tool_calls: bool = False):
        super().__init__(llm=llm, tools=tools, max_retries=max_retries,
                         return_trace=return_trace, stream=stream,
                         return_last_tool_calls=return_last_tool_calls)
        prompt = prompt or INSTRUCTION
        if self._return_last_tool_calls:
            prompt += '\nIf no more tool calls are needed, reply with ok and skip any summary.'
        self._assert_llm_tools()
        self._prompt = prompt
        self._agent = self.build_agent()

    def build_agent(self):
        return loop(FunctionCall(self._llm, self._tools, _prompt=self._prompt,
                                 return_trace=self._return_trace, stream=self._stream),
                    stop_condition=lambda x: isinstance(x, str), count=self._max_retries)

    def _pre_process(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        return (query, llm_chat_history or [])

    def _post_process(self, ret):
        if isinstance(ret, str):
            completed = self._pop_completed_tool_calls()
            if completed is not None:
                return completed
            return ret
        raise ValueError(f'After retrying {self._max_retries} times, the react agent still failes to call '
                         f'successfully.')
