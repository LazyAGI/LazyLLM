import copy
import json
from .formatterBase import LazyLLMFormatterBase as FormatterBase
import lazyllm
from typing import List, Dict, Union, Any

class JsonFormatter(FormatterBase):
    def _extract_json_from_string(self, mixed_str: str):
        json_objects = []
        brace_level = 0
        current_json = ""
        in_string = False

        for char in mixed_str:
            if char == '"' and (len(current_json) == 0 or current_json[-1] != '\\'):
                in_string = not in_string

            if not in_string:
                if char == '{':
                    if brace_level == 0:
                        current_json = ""
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1

            if brace_level > 0 or (brace_level == 0 and char == '}'):
                current_json += char

            if brace_level == 0 and current_json:
                try:
                    json.loads(current_json)
                    json_objects.append(current_json)
                    current_json = ""
                except json.JSONDecodeError:
                    continue

        return json_objects

    def _load_str(self, msg: str):
        # Convert str to json format
        assert msg.count("{") == msg.count("}"), f"{msg} is not a valid json string."
        try:
            json_strs = self._extract_json_from_string(msg)
            if len(json_strs) == 0:
                raise TypeError(f"{msg} is not a valid json string.")
            res = []
            for json_str in json_strs:
                res.append(json.loads(json_str))
            return res if len(res) > 1 else res[0]
        except Exception as e:
            lazyllm.LOG.info(f"Error: {e}")
            return ""

    def _parsing_format_output(self, keys: List, data: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if not keys:
            return data
        key = keys.pop(0)
        try:
            if isinstance(key, slice):
                return self._parsing_format_output(keys, data[key])
            elif isinstance(key, str):
                if isinstance(data, List):
                    res = [val[key] for val in data]
                    return self._parsing_format_output(keys, res if len(res) > 1 else res[0])
                elif isinstance(data, Dict):
                    return self._parsing_format_output(keys, data.get(key, {}))
                else:
                    return data
            else:
                raise TypeError(f"This class is not support {key} index.")
        except Exception as e:
            lazyllm.LOG.error(f"{e}")
            return ""

    def _parse_py_data_by_formatter(self, data):
        if self._slices is None:
            return data
        else:
            keys = copy.deepcopy(self._slices)
            result = self._parsing_format_output(keys, data)

            return result[0] if len(result) == 1 and isinstance(result, List) else result
