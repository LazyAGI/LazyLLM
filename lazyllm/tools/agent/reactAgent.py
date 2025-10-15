from lazyllm.module import ModuleBase
from lazyllm import loop, package, bind, pipeline
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
                 prompt: str = None, stream: bool = False, return_tool_call_results: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert llm and tools, 'llm and tools cannot be empty.'
        self._return_tool_call_results = return_tool_call_results
        with pipeline() as self._agent:
            with loop(stop_condition=lambda query, llm_chat_history, tool_call_results: isinstance(query, str),
                      count=self._max_retries) as react:
                react.pre_action = self._pre_action
                react.fc = FunctionCall(llm, tools, _prompt=prompt or INSTRUCTION, return_trace=return_trace,
                                        stream=stream)
                react.post_action = self._post_action | bind(llm_chat_history=react.input[0][1],
                                                             tool_call_results=react.input[0][2])
            self._agent.react = react
            self._agent.final_action = self._final_action

    def _pre_action(self, query: str, llm_chat_history: List[Dict[str, Any]], tool_call_results: List[str]):
        return package(query, llm_chat_history) if llm_chat_history is not None else query

    def _post_action(self, response: str, llm_chat_history: List[Dict[str, Any]], tool_call_results: List[str]):
        if self._return_tool_call_results and isinstance(response, list) and isinstance(response[-1], list):
            if len(response) >= 2 and response[-2].get('content', '') != '':
                tool_call_results.append(response[-2]['content'].strip())
            for item in response[-1]:
                if item.get('role', '') == 'tool':
                    tool_call_results.append(f'tool_name: {item.get("name", "")}, '
                                             f'arguments: {item.get("arguments", "")}, '
                                             f'result: {item.get("content", "")}')
        return package(response, llm_chat_history, tool_call_results)

    def _final_action(self, response: str, llm_chat_history: List[Dict[str, Any]], tool_call_results: List[str]):
        if self._return_tool_call_results:
            response = 'Tool call trace:\n' + '\n'.join(tool_call_results) + '\n\n' + response
        return response

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history, [])
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(ValueError(f'After retrying \
            {self._max_retries} times, the function call agent still failes to call successfully.'))
