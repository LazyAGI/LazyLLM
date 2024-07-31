from lazyllm.components import JsonFormatter
import json5 as json
from lazyllm import globals

class FunctionCallFormatter(JsonFormatter):
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
