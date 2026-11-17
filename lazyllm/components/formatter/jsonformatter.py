import json
from .formatterbase import JsonLikeFormatter
import lazyllm
from lazyllm.thirdparty import json_repair

class JsonFormatter(JsonLikeFormatter):
    def _extract_json_from_string(self, mixed_str: str):  # noqa: C901
        json_objects = []
        brace_level = 0
        current_json = ''
        in_string = False
        escaped = False

        for char in mixed_str:
            # Quotes only matter inside a JSON candidate; a stray `"` in surrounding prose must not flip the state.
            if in_string:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': in_string = False
            elif char == '"' and brace_level > 0:
                in_string = True

            if not in_string:
                if char in '{[':
                    if brace_level == 0:
                        current_json = ''
                    brace_level += 1
                elif char in '}]':
                    if brace_level == 0: continue  # stray closer in prose, not part of any JSON
                    brace_level -= 1

            if brace_level > 0 or (brace_level == 0 and char in '}]'):
                current_json += char

            if brace_level == 0 and current_json:
                try:
                    json_objects.append(json.loads(current_json))
                    current_json = ''
                except json.JSONDecodeError:
                    try:
                        repaired_obj = json_repair.loads(current_json)
                        json_objects.append(repaired_obj)
                        current_json = ''
                    except (json.JSONDecodeError, ValueError):
                        continue

        return json_objects

    def _load(self, msg: str):
        # Convert str to json format
        try:
            json_objects = self._extract_json_from_string(msg)
            if len(json_objects) == 0:
                raise TypeError(f'{msg} is not a valid json string.')
            return json_objects if len(json_objects) > 1 else json_objects[0]
        except Exception as e:
            lazyllm.LOG.info(f'Error: {e}')
            return ''
