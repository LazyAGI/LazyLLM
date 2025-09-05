# coding: utf-8

from dataclasses import dataclass
from typing import ClassVar, List


FC_PROMPT_LOCAL = """# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs:

{tool_start_token}The tool to use, should be one of tools list.
{tool_args_token}The input of the tool. The output format is: {"input1": param1, "input2": param2}. Can only return json.
{tool_end_token}End of tool."""


FC_PROMPT_ONLINE = ("Don't make assumptions about what values to plug into functions."
                    "Ask for clarification if a user request is ambiguous.\n")


@dataclass
class FunctionCallPrompts:
    """Prompt configuration class for Function Call operations"""

    local: str = FC_PROMPT_LOCAL
    online: str = FC_PROMPT_ONLINE

    KEYWORDS_IN_LOCAL: ClassVar[List[str]] = ["{tool_start_token}", "{tool_args_token}", "{tool_end_token}"]

    def __post_init__(self):
        if not isinstance(self.local, str):
            raise ValueError("local prompt must be a string")
        if not isinstance(self.online, str):
            raise ValueError("online prompt must be a string")

        def check_keywords(template: str, keywords: List[str]):
            for keyword in keywords:
                if keyword not in template:
                    raise ValueError(f"template must contain {keyword}")

        check_keywords(self.local, self.KEYWORDS_IN_LOCAL)
