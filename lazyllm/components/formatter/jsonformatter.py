import json
from .formatterbase import JsonLikeFormatter
import lazyllm
import json_repair

class JsonFormatter(JsonLikeFormatter):
    def _extract_json_from_string(self, mixed_str: str):  # noqa: C901
        json_objects = []
        brace_level = 0
        current_json = ''
        in_string = False

        for char in mixed_str:
            if char == '"' and (len(current_json) == 0 or current_json[-1] != '\\'):
                in_string = not in_string

            if not in_string:
                if char in '{[':
                    if brace_level == 0:
                        current_json = ''
                    brace_level += 1
                elif char in '}]':
                    brace_level -= 1

            if brace_level > 0 or (brace_level == 0 and char in '}]'):
                current_json += char

            if brace_level == 0 and current_json:
                try:
                    json.loads(current_json)
                    json_objects.append(current_json)
                    current_json = ''
                except json.JSONDecodeError:
                    try:
                        repaired_obj = json_repair.loads(current_json)
                        json.dumps(repaired_obj)
                        json_objects.append(repaired_obj)
                        current_json = ''
                    except (json.JSONDecodeError, TypeError, ValueError, Exception):
                        continue

        return json_objects

    def _load(self, msg: str):
        # Convert str to json format
        assert msg.count('{') == msg.count('}'), f'{msg} is not a valid json string.'
        try:
            json_strs = self._extract_json_from_string(msg)
            if len(json_strs) == 0:
                raise TypeError(f'{msg} is not a valid json string.')
            res = []
            for json_str in json_strs:
                res.append(json_str)
            return res if len(res) > 1 else res[0]
        except Exception as e:
            lazyllm.LOG.info(f'Error: {e}')
            return ''
