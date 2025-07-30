from lazyllm.components import JsonFormatter
import json5 as json
from lazyllm import globals

class FunctionCallFormatter(JsonFormatter):
    """Formatter for parsing structured function call messages.

This class extends `JsonFormatter` and is responsible for extracting JSON-based tool call structures from a mixed message string, optionally separating them using a global delimiter.

Private Method:
    _load(msg)
        Parses the input message string and extracts JSON-formatted tool calls, if present.


Examples:
    >>> from lazyllm.components import FunctionCallFormatter
    >>> formatter = FunctionCallFormatter()
    >>> msg = "Please call this tool. <TOOL> [{\"name\": \"search\", \"args\": {\"query\": \"weather\"}}]"
    >>> result = formatter._load(msg)
    >>> print(result)
    ... [{'name': 'search', 'args': {'query': 'weather'}}, 'Please call this tool. ']
    """
    def _load(self, msg: str):
        if "{" not in msg:
            return msg
        if globals['tool_delimiter'] in msg:
            content, msg = msg.split(globals['tool_delimiter'])
            assert msg.count("{") == msg.count("}"), f"{msg} is not a valid json string."
            try:
                json_strs = json.loads(msg)
                res = []
                for json_str in json_strs:
                    res.append(json_str)
                if content:
                    res.append(content)
                return res
            except Exception:
                return msg

        return msg
