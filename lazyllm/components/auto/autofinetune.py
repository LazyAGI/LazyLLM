import lazyllm
from lazyllm import launchers, finetune
from ..finetune.base import LazyLLMFinetuneBase
from .configure import get_configer
from .auto_helper import model_map, get_model_name, get_configs, check_requirements


class AutoFinetune(LazyLLMFinetuneBase):
    def __new__(cls, base_model, target_path, merge_path=None, ctx_len=1024,
                batch_size=32, lora_r=8, launcher=launchers.remote(), **kw):
        model_name = get_model_name(base_model)
        map_name = model_map(model_name)
        base_name = model_name.split('-')[0].split('_')[0].lower()
        candidates = get_configer().query_finetune(lazyllm.config['gpu_type'], launcher.ngpus,
                                                   map_name, ctx_len, batch_size, lora_r)
        configs = get_configs(base_name)

        for k, v in configs.items():
            if k not in kw: kw[k] = v

        for c in candidates:
            if check_requirements(c.framework.lower()):
                finetune_cls = getattr(finetune, c.framework.lower())
                for key, value in finetune_cls.auto_map.items():
                    if value:
                        kw[value] = getattr(c, key)
                return finetune_cls(base_model, target_path, merge_path, cp_files='tokeniz*',
                                    batch_size=batch_size, lora_r=lora_r, launcher=launcher, **kw)
        raise RuntimeError(f'No valid framework found, candidates are {[c.framework.lower() for c in candidates]}')
