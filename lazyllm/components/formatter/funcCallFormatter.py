import json
from json import JSONDecodeError
from .formatterBase import LazyLLMFormatterBase as FormatterBase
from typing import Any, Dict

class FunctionCallFormatter(FormatterBase):
    def _parse_py_data_by_formatter(self, data: Dict[str, Any]):
        isFC = False
        func_call = []
        if isinstance(data, dict):
            tool_calls = data.get("tool_calls", {})
            if "function_call" in data:
                data.pop("function_call")

            if tool_calls:
                isFC = True
                # Triggered the function call
                for tool_call in tool_calls:
                    tool_call_id = tool_call['id']
                    tool_name = tool_call['function']['name']
                    try:
                        if len(tool_call['function']['arguments'].strip()) == 0:
                            # This function call contains no parameters.
                            tool_input = {}
                        else:
                            tool_input = json.loads(tool_call['function']['arguments'], strict=False)
                    except JSONDecodeError:
                        raise TypeError(f"Cannot interpret tool_call {tool_call} \
                                          since the `arguments` is not valid JSON.")

                    func_call.append({"tool_call_id": tool_call_id, "name": tool_name, "tool_input": tool_input})

            return (isFC, data, func_call)

        else:
            raise TypeError(f"function call formatter does not support type {type(data)}")
