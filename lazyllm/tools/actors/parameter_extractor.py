from typing import Union
import json
from ...module import ModuleBase, TrainableModule, OnlineChatModuleBase
from ...common import package
import re

ch_parameter_extractor_prompt = """
你是一个智能助手，你的任务是从用户的输入中提取参数，并将其转换为json格式。
## 你需要根据提供的参数说明，从文本提取相应的内容生成json。参数说明如下：
'''{prompt}'''
其中name是参数的名称，type是参数的类型，description是参数的说明
## 提取要求如下
1. 你需要从用户的输入中提取出这些参数，并将其转换为json格式。请注意，json格式的键是参数的名称，值是提取到的值。请确保json格式正确，并且所有参数都被提取出来。
2. 如果用户的输入中没有提到某个参数，请将其值设置为null；如果在提取要求中设置了"require"：true，却没有提取到该字段，请在返回的json中设置__is_success为0，__reason设置失败原因。
__is_success字段解释：表示是否成功，成功时值为 1，失败时值为 0。
3. 若提供的参数说明为：[{{"name": "year", "type": "int", "description": "年份","require"：true}}]，你要用以下的json格式来返回结果：
{{"year": 2023,"__is_success": 1,"__reason":"提取成功"}}
4. 仅输出json字符串，输出的json字符串需要使用双引号，冒号使用英文冒号
5. 结合用户的输入和参数说明，尽量提取出更多的参数。
"""

en_parameter_extractor_prompt = """
You are an intelligent assistant. Your task is to extract parameters from the user’s input
and convert them into JSON.
You need to extract the following parameters from the text and generate JSON:
'''{prompt}'''
Each parameter has a name, type, description, and a "require" flag.
## Extraction requirements
1. Extract the specified parameters from the user’s input and convert them into JSON.
   The JSON keys should be the parameter names, and the values should be the extracted values.
   Ensure the JSON is valid and includes all parameters.
2. If a parameter is not mentioned in the input, set its value to null.
   If a parameter is marked as "require": true but is not found, set "__is_success" to 0
   and "__reason" to the failure description.
   - __is_success: indicates whether extraction succeeded (1) or failed (0).
3. For the provided parameter description:
   [{{"name": "year", "type": "int", "description": "Year", "require": true}}]
   you should return results in this format:
   {{"year": 2023, "__is_success": 1, "__reason": "Extraction successful"}}
4. Output only the JSON string—no other content.
   Use double quotes for JSON keys and English colons.
5. Use the user’s input and the parameter descriptions to extract as many parameters as possible.
"""

class ParameterExtractor(ModuleBase):
    type_map = {
        int.__name__: int,
        str.__name__: str,
        float.__name__: float,
        bool.__name__: bool,
        list.__name__: list,
        dict.__name__: dict,
    }

    def __init__(
        self,
        base_model: Union[str, TrainableModule, OnlineChatModuleBase],
        param: list[str],
        type: list[str],
        description: list[str],
        require: list[bool],
    ):
        super().__init__()
        assert len(param) == len(type) == len(description) == len(require) > 0
        self._param_dict = {p: ParameterExtractor.type_map[t] for p, t in zip(param, type)}
        param_prompt = repr([dict(name=p, type=t, description=d, require=r)
                             for p, t, d, r in zip(param, type, description, require)])
        self._prompt = self.choose_prompt(param_prompt).format(prompt=param_prompt)
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(self._prompt)
        else:
            self._m = base_model.share(self._prompt)

    def choose_prompt(self, prompt: str):
        # Use chinese prompt if intent elements have chinese character, otherwise use english version
        for ele in prompt:
            # chinese unicode range
            if "\u4e00" <= ele <= "\u9fff":
                return ch_parameter_extractor_prompt
        return en_parameter_extractor_prompt

    def forward(self, *args, **kw):
        res = self._m(*args, **kw)
        pattern = r"```json(.*?)\n```"
        matches = re.findall(pattern, res, re.DOTALL)
        if len(matches) > 0:
            res = matches[0]
            res.strip()
            try:
                res = json.loads(res)
            except Exception:
                pass
        else:
            res = res.split("\n")
            for param in res:
                try:
                    res = json.loads(param)
                except Exception:
                    continue
                if isinstance(res, dict): break
        if isinstance(res, dict):
            ret = [res.get(param_name, None) for param_name in self._param_dict]
        else:
            ret = [None] * len(self._param_dict)
        ret = package(ret)
        return ret
