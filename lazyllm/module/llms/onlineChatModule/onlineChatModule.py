import lazyllm
from typing import Any, Dict
from .openaiModule import OpenAIModule
from .glmModule import GLMModule
from .kimiModule import KimiModule
from .sensenovaModule import SenseNovaModule
from .qwenModule import QwenModule
from .doubaoModule import DoubaoModule
from .deepseekModule import DeepSeekModule
from .onlineChatModuleBase import OnlineChatModuleBase

class _ChatModuleMeta(type):

    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, OnlineChatModuleBase):
            return True
        return super().__instancecheck__(__instance)


class OnlineChatModule(metaclass=_ChatModuleMeta):
    """用来管理创建目前市面上公开的大模型平台访问模块，目前支持openai、sensenova、glm、kimi、qwen、doubao、deekseek(由于该平台暂时不让充值了，暂时不支持访问)。平台的api key获取方法参见 [开始入门](/#platform)

Args:
    model (str): 指定要访问的模型 (注意使用豆包时需用 Model ID 或 Endpoint ID，获取方式详见 [获取推理接入点](https://www.volcengine.com/docs/82379/1099522)。使用模型前，要先在豆包平台开通对应服务。)，默认为 ``gpt-3.5-turbo(openai)`` / ``SenseChat-5(sensenova)`` / ``glm-4(glm)`` / ``moonshot-v1-8k(kimi)`` / ``qwen-plus(qwen)`` / ``mistral-7b-instruct-v0.2(doubao)`` 
    source (str): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` / ``deepseek(暂时不支持访问)``
    base_url (str): 指定要访问的平台的基础链接，默认是官方链接
    system_prompt (str): 指定请求的system prompt，默认是官方给的system prompt
    stream (bool): 是否流式请求和输出，默认为流式
    return_trace (bool): 是否将结果记录在trace中，默认为False


Examples:
    >>> import lazyllm
    >>> from functools import partial
    >>> m = lazyllm.OnlineChatModule(source="sensenova", stream=True)
    >>> query = "Hello!"
    >>> with lazyllm.ThreadPoolExecutor(1) as executor:
    ...     future = executor.submit(partial(m, llm_chat_history=[]), query)
    ...     while True:
    ...         if value := lazyllm.FileSystemQueue().dequeue():
    ...             print(f"output: {''.join(value)}")
    ...         elif future.done():
    ...             break
    ...     print(f"ret: {future.result()}")
    ...
    output: Hello
    output: ! How can I assist you today?
    ret: Hello! How can I assist you today?
    >>> from lazyllm.components.formatter import encode_query_with_filepaths
    >>> vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-Vision")
    >>> query = "what is it?"
    >>> inputs = encode_query_with_filepaths(query, ["/path/to/your/image"])
    >>> print(vlm(inputs))
    """
    MODELS = {'openai': OpenAIModule,
              'sensenova': SenseNovaModule,
              'glm': GLMModule,
              'kimi': KimiModule,
              'qwen': QwenModule,
              'doubao': DoubaoModule,
              'deepseek': DeepSeekModule}

    @staticmethod
    def _encapsulate_parameters(base_url: str,
                                model: str,
                                stream: bool,
                                return_trace: bool,
                                **kwargs) -> Dict[str, Any]:
        params = {"stream": stream, "return_trace": return_trace}
        if base_url is not None:
            params['base_url'] = base_url
        if model is not None:
            params['model'] = model
        params.update(kwargs)

        return params

    def __new__(self,
                model: str = None,
                source: str = None,
                base_url: str = None,
                stream: bool = True,
                return_trace: bool = False,
                **kwargs):
        if model in OnlineChatModule.MODELS.keys() and source is None: source, model = model, source

        params = OnlineChatModule._encapsulate_parameters(base_url, model, stream, return_trace, **kwargs)

        if kwargs.get("skip_auth", False):
            source = source or "openai"
            if not base_url:
                raise KeyError("base_url must be set for local serving.")

        if source is None:
            if "api_key" in kwargs and kwargs["api_key"]:
                raise ValueError("No source is given but an api_key is provided.")
            for source in OnlineChatModule.MODELS.keys():
                if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f"No api_key is configured for any of the models {OnlineChatModule.MODELS.keys()}.")

        assert source in OnlineChatModule.MODELS.keys(), f"Unsupported source: {source}"
        return OnlineChatModule.MODELS[source](**params)
