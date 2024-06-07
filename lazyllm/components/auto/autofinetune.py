import lazyllm
from lazyllm import launchers, finetune
from ..finetune.base import LazyLLMFinetuneBase
from .configure import get_configer
from .auto_helper import model_map, get_model_name, get_configs, check_requirements
from ..utils.downloader import ModelDownloader


class AutoFinetune(LazyLLMFinetuneBase):
    def __new__(cls, base_model, target_path, source=lazyllm.config['model_source'], merge_path=None, ctx_len=1024,
                batch_size=32, lora_r=8, launcher=launchers.remote(ngpus=1), **kw):
        base_model = ModelDownloader(source).download(base_model)
        model_name = get_model_name(base_model)
        if cls.get_model_type(model_name) == 'embed':
            raise RuntimeError('Fine-tuning of the embed model is not currently supported.')
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

    @classmethod
    def get_model_type(cls, model_name):
        from lazyllm.components.utils.downloader.model_mapping import model_name_mapping
        if model_name in model_name_mapping:
            return model_name_mapping[model_name].get('type', 'llm')
        else:
            return 'llm'
