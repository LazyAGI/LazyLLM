import lazyllm
from lazyllm import launchers, finetune
from ..finetune.base import LazyLLMFinetuneBase
from .configure import get_configer
from .dependencies.requirements import requirements
from .auto_helper import model_map, get_model_name, get_configs, check_requirements
from ..utils.downloader import ModelManager


class AutoFinetune(LazyLLMFinetuneBase):
    """This class is a subclass of ``LazyLLMFinetuneBase`` and can automatically select the appropriate fine-tuning framework and parameters based on the input arguments to fine-tune large language models.

Specifically, based on the input model parameters of ``base_model``, ``ctx_len``, ``batch_size``, ``lora_r``, the type and number of GPUs in ``launcher``, this class can automatically select the appropriate fine-tuning framework (such as: ``AlpacaloraFinetune`` or ``CollieFinetune``) and the required parameters.

Args:
    base_model (str): The base model used for fine-tuning. It is required to be the path of the base model.
    source (lazyllm.config['model_source']): Specifies the model download source. This can be configured by setting the environment variable ``LAZYLLM_MODEL_SOURCE``.
    target_path (str): The path where the LoRA weights of the fine-tuned model are saved.
    merge_path (str): The path where the model merges the LoRA weights, default to ``None``. If not specified, "lazyllm_lora" and "lazyllm_merge" directories will be created under ``target_path`` as ``target_path`` and ``merge_path`` respectively.
    ctx_len (int): The maximum token length for input to the fine-tuned model, default to ``1024``.
    batch_size (int): Batch size, default to ``32``.
    lora_r (int): LoRA rank, default to ``8``; this value determines the amount of parameters added, the smaller the value, the fewer the parameters.
    launcher (lazyllm.launcher): The launcher for fine-tuning, default to ``launchers.remote(ngpus=1)``.
    kw: Keyword arguments, used to update the default training parameters. Note that additional keyword arguments cannot be arbitrarily specified, as they depend on the framework inferred by LazyLLM, so it is recommended to set them with caution.



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
