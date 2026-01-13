import os
from enum import Enum
import json
import threading
from typing import List, Union, Optional
from contextlib import contextmanager


class Mode(Enum):
    Display = 0,
    Normal = 1,
    Debug = 2,


class _ConfigMeta(type):
    _registered_cfgs = dict()
    _env_name_map = dict()
    _homes = dict()
    _doc = ''
    _instances = dict()
    _lock = threading.RLock()

    def __call__(cls, prefix: str = 'LAZYLLM', *args, **kwargs):
        if prefix.lower() not in cls._instances:
            with cls._lock:
                if prefix.lower() not in cls._instances:
                    cls._instances[prefix.lower()] = super().__call__(prefix, *args, **kwargs)
        return cls._instances[prefix.lower()]

    @staticmethod
    def _get_description(name):
        desc = _ConfigMeta._registered_cfgs[name]
        if not desc: raise ValueError(f'Description for {name} is not found')
        doc = (f'  - Description: {desc["description"]}, type: `{desc["type"].__name__}`, '
               'default: `{desc["default"]}`<br>\n')
        if (options := desc.get('options')):
            doc += f'  - Options: {", ".join(options)}<br>\n'
        if (env := desc.get('env')):
            if isinstance(env, str):
                doc += f'  - Environment Variable: {("LAZYLLM_" + env).upper()}<br>\n'
            elif isinstance(env, dict):
                doc += '  - Environment Variable:<br>\n'
                for k, v in env.items():
                    doc += f'{("    - LAZYLLM_" + k).upper()}: {v}<br>\n'
        return doc

    @staticmethod
    def add(name: str, type: type, default: Optional[Union[int, str, bool]] = None, env: Union[str, dict] = None,
            *, options: Optional[List] = None, description: Optional[str] = None):
        update_params = dict(type=type, default=default, env=env, options=options, description=description)
        if name not in _ConfigMeta._registered_cfgs or _ConfigMeta._registered_cfgs[name] != update_params:
            _ConfigMeta._registered_cfgs[name] = update_params
            if not env: env = name.lower()
            for k in ([env] if isinstance(env, str) else env.keys() if isinstance(env, dict) else env):
                _ConfigMeta._env_name_map[k.lower()] = name
        for v in _ConfigMeta._instances.values():
            v._update_impl(name, type, default, env)

    def _get_default_home(prefix):
        return _ConfigMeta._homes[prefix]

    @property
    def __doc__(self):
        doc = f'{self._doc}\n**LazyLLM Configurations:**\n\n'
        return doc + '<br>\n'.join([f'- **{name}**:<br>\n{self._get_description(name)}'
                                    for name in self._registered_cfgs.keys()])

    @__doc__.setter
    def __doc__(self, value):
        self._doc = value

    def __contains__(self, key):
        return key.lower() in self._instances or '_' in key and key.split('_')[0].lower() in self._instances


_ConfigMeta.add('home', str, None, 'HOME', description='The default home directory for LazyLLM.')

class Config(metaclass=_ConfigMeta):
    def __init__(self, prefix: str = 'LAZYLLM', home: Optional[str] = None, config_file: str = 'config.json'):
        self._prefix = prefix.upper()
        self._impl, self._cfgs = dict(), dict()
        if not home:
            home = '.lazyllm' if self._prefix == 'LAZYLLM' else f'.lazyllm_{prefix.lower()}'
            home = os.path.join(os.path.expanduser('~'), home)
        _ConfigMeta._homes[self._prefix] = home
        self._update_impl('home', str, None, 'HOME')
        os.makedirs(self['home'], exist_ok=True)
        self._cgf_path = os.path.join(self['home'], config_file)
        if os.path.exists(self._cgf_path):
            with open(self._cgf_path, 'r+') as f:
                self._cfgs = Config.get_config(json.loads(f))
        for name, cfg in _ConfigMeta._registered_cfgs.items():
            if name == 'home': continue
            self._update_impl(name, cfg['type'], cfg['default'], cfg['env'])

    def add(self, name: str, type: type, default: Optional[Union[int, str, bool]] = None, env: Union[str, dict] = None,
            *, options: Optional[List] = None, description: Optional[str] = None):
        if name in Config.__dict__.keys():
            raise RuntimeError(f'{name} is attribute of Config, please change it!')
        _ConfigMeta.add(name, type, default, env, options=options, description=description)

    def getenv(self, name, type, default=None):
        r = os.getenv(f'{self._prefix}_{name.upper()}', default)
        if type == bool:
            return r in (True, 'TRUE', 'True', 1, 'ON', '1')
        return type(r) if r is not None else r

    @staticmethod
    def get_config(cfg):
        return cfg

    def get_all_configs(self):
        return self._impl

    def done(self):
        ins = _ConfigMeta._instances['lazyllm']
        assert len(ins._cfgs) == 0, f'Invalid cfgs ({"".join(ins._cfgs.keys())}) are given in {ins._cgf_path}'

    @contextmanager
    def temp(self, name, value):
        old_value = self[name]
        self._impl[name] = value
        yield
        self._impl[name] = old_value

    def _update_impl(self, name: str, type: type, default: Optional[Union[int, str, bool]] = None,
                     env: Union[str, dict, list] = None):
        self._impl[name] = self._cfgs.pop(name) if name in self._cfgs else (
            _ConfigMeta._get_default_home(self._prefix) if name == 'home' else default)
        if isinstance(env, dict):
            for k, v in env.items():
                if self.getenv(k, bool):
                    self._impl[name] = v
                    break
        elif isinstance(env, list):
            for k in env:
                if (v := self.getenv(k, type)):
                    self._impl[name] = v
                    break
        elif env:
            self._impl[name] = self.getenv(env, type, self._impl[name])
        if not isinstance(self._impl[name], type) and self._impl[name] is not None: raise TypeError(
            f'Invalid config type for {name}, type is {type}')

    def __getitem__(self, name):
        try:
            if isinstance(name, bytes): name = name.decode('utf-8')
            name = name.lower()
            if name.startswith(f'{self._prefix.lower()}_'): name = name[len(self._prefix) + 1:]
            return self._impl[name]
        except KeyError as e:
            raise KeyError(f'Error occured when getting key `{name}` from lazyllm global config, msg is: {e}')

    def __str__(self):
        return str(self._impl)

    @property
    def _envs(self):
        return [f'{self._prefix}_{e}'.lower() for e in _ConfigMeta._env_name_map.keys()]

    def refresh(self, targets: Union[bytes, str, List[str]] = None) -> None:
        names, all_envs = targets, self._envs
        if isinstance(targets, bytes): targets = targets.decode('utf-8')
        if isinstance(targets, str): names = [targets.lower()]
        elif targets is None:
            names = [key.lower() for key in os.environ.keys() if key.lower() in all_envs]
        assert isinstance(names, list)
        for name in names:
            if name.lower() in all_envs:
                name = _ConfigMeta._env_name_map[name[len(self._prefix) + 1:]]
            elif name in Config: continue
            cfg = _ConfigMeta._registered_cfgs[name]
            if name in self._impl: self._update_impl(name, cfg['type'], cfg['default'], cfg['env'])


class _NamespaceConfig(object):
    def __init__(self):
        self._config = Config()

    @property
    def _impl(self): return self._config._impl

    def add(self, name: str, type: type, default: Optional[Union[int, str, bool]] = None, env: Union[str, dict] = None,
            *, options: Optional[List] = None, description: Optional[str] = None):
        if name in Config.__dict__.keys() or name in _NamespaceConfig.__dict__.keys():
            raise RuntimeError(f'{name} is attribute of Config, please change it!')
        _ConfigMeta.add(name, type, default, env, options=options, description=description)
        return self

    def __getitem__(self, __key):
        return self._config[__key]

    def __getattr__(self, __key: str):
        try:
            return self[__key]
        except KeyError:
            raise AttributeError(f'Config has no attribute {__key}')

    def refresh(self, targets: Union[bytes, str, List[str]] = None) -> None:
        return self._config.refresh(targets)

    def get_all_configs(self):
        return self._config._impl

    @contextmanager
    def temp(self, name, value):
        with self._config.temp(name, value):
            yield

    def done(self):
        return self._config.done()


config = _NamespaceConfig().add('mode', Mode, Mode.Normal, dict(DISPLAY=Mode.Display, DEBUG=Mode.Debug),
                                description='The default mode for LazyLLM.'
        ).add('repr_ml', bool, False, 'REPR_USE_ML', description='Whether to use Markup Language for repr.'
        ).add('repr_show_child', bool, False, 'REPR_SHOW_CHILD',
              description='Whether to show child modules in repr.'
        ).add('rag_store', str, 'none', 'RAG_STORE', description='The default store for RAG.'
        ).add('gpu_type', str, 'A100', 'GPU_TYPE', description='The default GPU type for LazyLLM.'
        ).add('train_target_root', str, os.path.join(os.getcwd(), 'save_ckpt'), 'TRAIN_TARGET_ROOT',
              description='The default target root for training.'
        ).add('infer_log_root', str, os.path.join(os.getcwd(), 'infer_log'), 'INFER_LOG_ROOT',
              description='The default log root for inference.'
        ).add('temp_dir', str, os.path.join(os.getcwd(), '.temp'), 'TEMP_DIR',
              description='The default temp directory for LazyLLM.'
        ).add('thread_pool_worker_num', int, 16, 'THREAD_POOL_WORKER_NUM',
              description='The default number of workers for thread pool.'
        ).add('deploy_skip_check_kw', bool, False, 'DEPLOY_SKIP_CHECK_KW',
              description='Whether to skip check keywords for deployment.'
        ).add('allow_internal_network', bool, False, 'ALLOW_INTERNAL_NETWORK',
              description='Whether to allow loading images from internal network addresses. '
                          'Set to False for security in production environments.')

def refresh_config(key):
    if key in Config:
        Config._instances[key.split('_')[0].lower()].refresh(key)
