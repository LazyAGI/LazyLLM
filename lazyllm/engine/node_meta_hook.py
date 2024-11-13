import time
import requests
import json
import lazyllm
from lazyllm import globals, LazyLLMHook


class MetaKeys:
    ID: str = "id"
    SESSIONID: str = "sessionid"
    TIMECOST: str = "timecost"
    PROMPT_TOKENS: str = "prompt_tokens"
    COMPLETION_TOKENS: str = "completion_tokens"
    INPUT: str = "input"
    OUTPUT: str = "output"


class NodeMetaHook(LazyLLMHook):

    def __init__(self, obj, url, front_id):
        if isinstance(obj, lazyllm.ModuleBase):
            self._uniqueid = obj._module_id
        elif isinstance(obj, lazyllm.FlowBase):
            self._uniqueid = obj._flow_id
        else:
            raise TypeError(f"Expected 'obj' to be type of ModuleBase or FlowBase, but got {type(obj)}")
        self._meta_info = {
            MetaKeys.ID: str(self._uniqueid),
            MetaKeys.TIMECOST: 0.0,
            MetaKeys.PROMPT_TOKENS: 0,
            MetaKeys.COMPLETION_TOKENS: 0,
            MetaKeys.INPUT: "",
            MetaKeys.OUTPUT: "",
        }
        self._front_id = front_id
        self._url = url

    def pre_hook(self, *args, **kwargs):
        arguments = {}
        self._meta_info[MetaKeys.SESSIONID] = lazyllm.globals._sid
        if len(args) == 1:
            if isinstance(args[0], lazyllm.package) and len(args[0]) == 1:
                self._meta_info[MetaKeys.INPUT] = str(args[0][0])
            else:
                self._meta_info[MetaKeys.INPUT] = str(args[0])
        else:
            for index, value in enumerate(args):
                arguments[f"arg_{index}"] = value
            arguments.update(kwargs)
            self._meta_info[MetaKeys.INPUT] = str(arguments)
        self._meta_info[MetaKeys.TIMECOST] = time.time()

    def post_hook(self, output):
        self._meta_info[MetaKeys.OUTPUT] = str(output)

        if self._uniqueid in globals["usage"]:
            self._meta_info.update(globals["usage"][self._uniqueid])
        self._meta_info[MetaKeys.ID] = self._front_id
        self._meta_info[MetaKeys.TIMECOST] = time.time() - self._meta_info[MetaKeys.TIMECOST]

    def report(self):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        json_data = json.dumps(self._meta_info, ensure_ascii=False)
        try:
            requests.post(self._url, data=json_data, headers=headers)
        except Exception as e:
            lazyllm.LOG.warning(f"Error sending collected data: {e}. URL: {self._url}")
