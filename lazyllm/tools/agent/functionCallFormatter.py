from lazyllm.components import JsonFormatter
from lazyllm import json5 as json

class FunctionCallFormatter(JsonFormatter):
    def _load(self, msg: str):
        if "{" not in msg:
            return msg
        if "<|tool_calls|>" in msg:
            content, msg = msg.split("<|tool_calls|>")
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
