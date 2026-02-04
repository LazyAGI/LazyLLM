import lazyllm
from typing import Tuple, List, Dict, Any, Optional 
from ..base_data import data_register

def llm_serving(if_online:bool, user_prompt:str, system_prompt:str, model_name:Optional[str]='Qwen2.5-0.5B-Instruct', **kwargs) -> str:
    prompter = lazyllm.ChatPrompter(system_prompt)
    if if_online:
        api_key = kwargs.get('api_key')
        if api_key:
            llm = lazyllm.OnlineChatModule(model=model_name, api_key=api_key).prompt(prompter)
        else:
            raise ValueError("onlinemodule needs api_key")
    else:
        llm_path = kwargs.get('llm_path', "")
        llm = lazyllm.TrainableModule(base_model=model_name, target_path=llm_path).prompt(prompter)
        llm.start()
    result = llm(user_prompt)
    return result