from typing import List, Dict, Any, Union
from lazyllm.components.prompter import LazyLLMPrompterBase
from overrides import override
import copy

from .protocol import ROLE, SYSTEM

class AgentPrompter(LazyLLMPrompterBase):
    def __init__(self, instruction: None | str = None, extro_keys: None | List[str] = None, show: bool = False):
        super().__init__(show, tools=None)
        instruction = instruction or ""
        instruction_template = f'{instruction}\n{{extro_keys}}\n'.replace('{extro_keys}', LazyLLMPrompterBase._get_extro_key_template(extro_keys))
        self._init_prompt("{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{input}\n{eoh}{soa}\n",
                          instruction_template)

    @property
    def _split(self): return self._soa

    @override
    def _generate_prompt_dict_impl(self, 
        instruction:str = None,
        input:Union[str, Dict] = None, 
        history:List[Dict[str, Any]] = None, 
        tools:List[Dict[str, Any]] = None, 
        label:str = None) -> Dict[str, Any]:
        """
        生成请求 openai api 格式的 request data, 只包含messages和tools

        Args:
            instruction (str): 跟随在system_message后的指令
            input (Union[str, Dict]): 最后一条消息，在此处不使用该参数
            history (List[Dict[str, Any]]): 历史对话数据, 包括用户最近的一次输入或最近的N条TOOL的输出, 一般不为空
            tools (List[Dict[str, Any]]): 要调用工具的描述
            label (str): 追加在最后一条消息后的字符串, 此处不使用该参数
        """
        assert history is not None, "history is None"
        messages = copy.deepcopy(history)
        if messages[0][ROLE] != SYSTEM:
            messages.insert(0, dict(role=SYSTEM, content=self._system + "\n" + instruction if instruction else self._system))
        return dict(messages=messages, tools=tools) if tools else dict(messages=messages)