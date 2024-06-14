import json
from .formatterBase import LazyLLMFormatterBase as FormatterBase
import lazyllm
from typing import List, Dict

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

    def _str_to_python(self, msg: str):
        # Convert str to json format
        assert msg.count("{") == msg.count("}")
        try:
            json_strs = self._extract_json_from_string(msg)
            res = []
            for json_str in json_strs:
                res.append(json.loads(json_str))
            return res if len(res) > 1 else res[0]
        except Exception as e:
            lazyllm.LOG.info(f"Error: {e}")
            return ""

    def _parse_py_data_by_formatter(self, data):
        if self._slices is None:
            return data
        else:
            result = data
            try:
                for s in self._slices:
                    if isinstance(s, slice):
                        result = result[s]
                    elif isinstance(s, str):
                        if isinstance(result, List):
                            res = [val[s] for val in result]
                            result = res if len(res) > 1 else res[0]
                        elif isinstance(result, Dict):
                            result = result[s]
                        else:
                            raise TypeError(f"{result} is not support {s} index.")
                    else:
                        raise TypeError(f"This class is not support {s} index.")
            except Exception as e:
                lazyllm.LOG.error(f"{e}")
                return ""

            return result
