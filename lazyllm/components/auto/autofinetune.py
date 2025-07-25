import lazyllm
from lazyllm import launchers, finetune
from ..finetune.base import LazyLLMFinetuneBase
from .configure import get_configer
from .dependencies.requirements import requirements
from .auto_helper import model_map, get_model_name, get_configs, check_requirements
from ..utils.downloader import ModelManager


class AutoFinetune(LazyLLMFinetuneBase):
    """此类是 ``LazyLLMFinetuneBase`` 的子类，可根据输入的参数自动选择合适的微调框架和参数，以对大语言模型进行微调。

具体而言，基于输入的：``base_model`` 的模型参数、``ctx_len``、``batch_size``、``lora_r``、``launcher`` 中GPU的类型以及卡数，该类可以自动选择出合适的微调框架（如: ``AlpacaloraFinetune`` 或 ``CollieFinetune``）及所需的参数。

Args:
    base_model (str): 用于进行微调的基模型。要求是基模型的路径。
    source (lazyllm.config['model_source']): 指定模型的下载源。可通过设置环境变量 ``LAZYLLM_MODEL_SOURCE`` 来配置，目前仅支持 ``huggingface`` 或 ``modelscope`` 。若不设置，lazyllm不会启动自动模型下载。
    target_path (str): 微调后模型保存LoRA权重的路径。
    merge_path (str): 模型合并LoRA权重后的路径，默认为 ``None``。如果未指定，则会在 ``target_path`` 下创建 "lazyllm_lora" 和 "lazyllm_merge" 目录，分别作为 ``target_path`` 和  ``merge_path`` 。
    ctx_len (int): 输入微调模型的token最大长度，默认为 ``1024``。
    batch_size (int): 批处理大小，默认为 ``32``。
    lora_r (int): LoRA 的秩，默认为 ``8``；该数值决定添加参数的量，数值越小参数量越小。
    launcher (lazyllm.launcher): 微调的启动器，默认为 ``launchers.remote(ngpus=1)``。
    kw: 关键字参数，用于更新默认的训练参数。注意这里能够指定的关键字参数取决于 LazyLLM 推测出的框架，因此建议谨慎设置。



Examples:
    >>> from lazyllm import finetune
    >>> finetune.auto("internlm2-chat-7b", 'path/to/target')
    <lazyllm.llm.finetune type=AlpacaloraFinetune>
    """
    def __new__(cls, base_model, target_path, source=lazyllm.config['model_source'], merge_path=None, ctx_len=1024,
                batch_size=32, lora_r=8, launcher=launchers.remote(ngpus=1), **kw):
        base_model = ModelManager(source).download(base_model) or ''
        model_name = get_model_name(base_model)
        model_type = ModelManager.get_model_type(model_name)
        if model_type in ['embed', 'tts', 'vlm', 'stt', 'sd']:
            raise RuntimeError(f'Fine-tuning of the {model_type} model is not currently supported.')
        map_name, _ = model_map(model_name)
        base_name = model_name.split('-')[0].split('_')[0].lower()
        candidates = get_configer().query_finetune(lazyllm.config['gpu_type'], launcher.ngpus,
                                                   map_name, ctx_len, batch_size, lora_r)
        configs = get_configs(base_name)

        for k, v in configs.items():
            if k not in kw: kw[k] = v

        for c in candidates:
            if check_requirements(requirements[c.framework.lower()]):
                finetune_cls = getattr(finetune, c.framework.lower())
                for key, value in finetune_cls.auto_map.items():
                    if value:
                        kw[value] = getattr(c, key)
                if finetune_cls.__name__ == 'LlamafactoryFinetune':
                    return finetune_cls(base_model, target_path, lora_r=lora_r, launcher=launcher, **kw)
                return finetune_cls(base_model, target_path, merge_path, cp_files='tokeniz*',
                                    batch_size=batch_size, lora_r=lora_r, launcher=launcher, **kw)
        raise RuntimeError(f'No valid framework found, candidates are {[c.framework.lower() for c in candidates]}')
