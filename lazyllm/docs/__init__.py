from .utils import add_doc
from lazyllm import config

config.add('init_doc', bool, False, 'INIT_DOC', description='whether to init docs')
if config['init_doc'] and (add_doc.__doc__ is None or 'Add document' not in add_doc.__doc__):
    from . import common, components, configs, flow, hook, launcher, module, patch, prompt_template, tools, utils  # noqa F401
    del common, components, configs, flow, hook, launcher, module, patch, prompt_template, tools, utils

__all__ = [
    'add_doc'
]
