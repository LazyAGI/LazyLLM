import time
import requests
import json
import lazyllm
from lazyllm import globals
from abc import ABC, abstractmethod


class MetaKeys:
    ID: str = "id"
    SESSIONID: str = "sessionid"
    TIMECOST: str = "timecost"
    PROMPT_TOKENS: str = "prompt_tokens"
    COMPLETION_TOKENS: str = "completion_tokens"
    INPUT: str = "input"
    OUTPUT: str = "output"


class LazyllmHook(ABC):
    @abstractmethod
    def __init__(self, obj):
        pass

    @abstractmethod
    def pre_hook(self, *args, **kwargs):
        pass

    @abstractmethod
    def post_hook(self, output):
        pass

    @abstractmethod
    def report():
        pass


class NodeMetaHook(LazyllmHook):
    URL = ""
    MODULEID_TO_WIDGETID = {}

    def __init__(self, obj):
        if isinstance(obj, lazyllm.ModuleBase):
            self._uniqueid = obj._module_id
        elif isinstance(obj, lazyllm.FlowBase):
            self._uniqueid = obj._flow_id
        else:
            raise TypeError(f"Expected 'obj' to be type of ModuleBase or FlowBase, but got {type(obj)}")
        self._meta_info = {
            MetaKeys.ID: str(self._uniqueid),
            MetaKeys.SESSIONID: lazyllm.globals._sid,
            MetaKeys.TIMECOST: 0.0,
            MetaKeys.PROMPT_TOKENS: 0,
            MetaKeys.COMPLETION_TOKENS: 0,
            MetaKeys.INPUT: "",
            MetaKeys.OUTPUT: "",
        }

    def pre_hook(self, *args, **kwargs):
        arguments = {}
        if len(args) == 1:
            self._meta_info[MetaKeys.INPUT] = str(args[0])
        else:
            for index, value in enumerate(args):
                arguments[f"arg_{index}"] = value
            arguments.update(kwargs)
            self._meta_info[MetaKeys.INPUT] = str(arguments)
        self._meta_info[MetaKeys.TIMECOST] = time.time()

    def post_hook(self, output):
        if isinstance(output, tuple):
            raise TypeError("Unexpected tuple for output in post_hook")
        self._meta_info[MetaKeys.OUTPUT] = str(output)

        if self._uniqueid in globals["usage"]:
            self._meta_info.update(globals["usage"])
        if self._meta_info[MetaKeys.ID] in self.MODULEID_TO_WIDGETID:
            self._meta_info[MetaKeys.ID] = self.MODULEID_TO_WIDGETID[self._meta_info[MetaKeys.ID]]
        self._meta_info[MetaKeys.TIMECOST] = time.time() - self._meta_info[MetaKeys.TIMECOST]

    def report(self):
        headers = {"Content-Type": "application/json; charset=utf-8"}
        json_data = json.dumps(self._meta_info, ensure_ascii=False)
        try:
            lazyllm.LOG.info(f"meta_info: {self._meta_info}")
            requests.post(self.URL, data=json_data, headers=headers)
        except Exception as e:
            lazyllm.LOG.warning(f"Error sending collected data: {e}")
