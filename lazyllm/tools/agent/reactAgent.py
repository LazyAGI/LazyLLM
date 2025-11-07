from lazyllm.module import ModuleBase
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


class ReactAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False,
                 prompt: str = None, stream: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert llm and tools, 'llm and tools cannot be empty.'
        self._agent = loop(FunctionCall(llm, tools, _prompt=prompt, return_trace=return_trace, stream=stream),
                           stop_condition=lambda x: len(x.get('tool_calls', [])) == 0, count=self._max_retries)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history) if llm_chat_history is not None else self._agent(query)
        if len(ret.get('tool_calls', [])) == 0: return ret['content']
        raise ValueError(
            f'After retrying {self._max_retries} times, the react agent still failed to call successfully.'
        )
