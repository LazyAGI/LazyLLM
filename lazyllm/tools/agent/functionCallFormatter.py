from lazyllm.components import JsonFormatter
import json

class FunctionCallFormatter(JsonFormatter):
    def _load(self, msg: str):
        if "{" not in msg:
            return msg
        is_tc = False
        if "<|tool_calls|>" in msg:
            is_tc = True
            msg = msg.split("<|tool_calls|>")[1]
            assert msg.count("{") == msg.count("}"), f"{msg} is not a valid json string."
        try:
            json_strs = self._extract_json_from_string(msg)
            if len(json_strs) == 0:
                return msg if not is_tc else (_ for _ in ()).throw(TypeError(f"{msg} is not a valid json string."))
            res = []
            for json_str in json_strs:
                res.append(json.loads(json_str))
            return res
        except Exception:
            return msg if not is_tc else (_ for _ in ()).throw(TypeError(f"{msg} is not a valid json string."))
