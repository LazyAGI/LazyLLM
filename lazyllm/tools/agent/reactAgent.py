from lazyllm.module import ModuleBase
from lazyllm import loop, LOG
from .toolsManager import ToolManager
from .functionCall import FunctionCall
from typing import List, Any, Dict, Union
import re

class ReactFunctionCall(FunctionCall):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False, _prompt: str = None):
        super().__init__(llm=llm, tools=tools, return_trace=return_trace, _prompt=_prompt)

    def _parser(self, llm_output: Union[str, List[Dict[str, Any]]]):
        LOG.info(f"llm_output: {llm_output}")
        if isinstance(llm_output, str):
            if "Action" in llm_output:
                match = re.search(r"Action: ([a-zA-Z0-9_]+).*?\n+Action Input: .*?(\{.*\})", llm_output, re.DOTALL)
                if match:
                    action = match.group(1).strip()
                    action_input = match.group(2).strip()
                    tools = [{'name': action, 'arguments': action_input}]
                    LOG.info(f"tools: {tools}")
                    return tools
                else:
                    raise ValueError(f"Tools and parameters do not conform to formatting requirements in {llm_output}.")
            elif "Answer" in llm_output:
                match = re.search(r"Answer:(.*?)(?:$)", llm_output, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    LOG.info(f"answer: {answer}")
                    return answer
                else:
                    raise ValueError(f"The text {llm_output} does not contain the final answer.")
            else:
                raise ValueError(f"Could not parse output: {llm_output}.")
        else:
            raise TypeError(f"Model output currently only supports the str type and does not support {type(llm_output)}")

    def _tool_post_action(self, output: Union[str, List[str]], input: Union[str, List],
                          llm_output: List[Dict[str, Any]]):
        LOG.info(f"tool output: {output}")
        if isinstance(output, list):
            ret = []
            if isinstance(input, str):
                ret.append(input)
            elif isinstance(input, list):
                ret.append(input[-1])
            else:
                raise TypeError(f"The input type currently only supports `str` and `list`, "
                                f"and does not support {type(input)}.")
            ret.append({"role": "assistant", "content": llm_output})
            ret.append([{"role": "user", "content": "Observation: " + out} for out in output])
            LOG.info(f"ret: {ret}")
            return ret
        elif isinstance(output, str):
            return output
        else:
            raise TypeError(f"The {output} type currently is only supports `list` and `str`,"
                            f" and does not support {type(output)}.")

INSTRUCTION = """You are designed to help with a variety of tasks, from answering questions to providing \
summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence \
you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", \
"num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using \
any more tools. At that point, you MUST respond in the one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages."""

class ReactAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert llm and tools, "llm and tools cannot be empty."
        self._tools_manager = ToolManager(tools)
        self._agent = loop(ReactFunctionCall(llm, tools, _prompt=INSTRUCTION.format(tool_desc=tools,
                                             tool_names=self._tools_manager.tools_description)),
                           stop_condition=lambda x: isinstance(x, str), count=self._max_retries)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history=[] if llm_chat_history is None else llm_chat_history)
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(ValueError(f"After retrying \
            {self._max_retries} times, the function call agent still failes to call successfully."))
