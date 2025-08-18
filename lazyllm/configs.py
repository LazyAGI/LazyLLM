import os
from enum import Enum
import json
from typing import List, Union, Optional
from contextlib import contextmanager
import logging


class Mode(Enum):
    Display = 0,
    Normal = 1,
    Debug = 2,


class Config(object):
    def __init__(self, prefix='LAZYLLM', home=os.path.join(os.path.expanduser('~'), '.lazyllm')):  # noqa B008
        self._config_params = dict()
        self._env_map_name = dict()
        self.prefix = prefix
        self.impl, self.cfgs = dict(), dict()
        self.add('home', str, os.path.expanduser(home), 'HOME')
        os.makedirs(home, exist_ok=True)
        self.cgf_path = os.path.join(self['home'], 'config.json')
        if os.path.exists(self.cgf_path):
            with open(self.cgf_path, 'r+') as f:
                self.cfgs = Config.get_config(json.loads(f))

    def done(self):
        assert len(self.cfgs) == 0, f'Invalid cfgs ({"".join(self.cfgs.keys())}) are given in {self.cgf_path}'
        return self

    def getenv(self, name, type, default=None):
        r = os.getenv(f'{self.prefix}_{name.upper()}', default)
        if type == bool:
            return r in (True, 'TRUE', 'True', 1, 'ON', '1')
        return type(r)

    @staticmethod
    def get_config(cfg):
        return cfg

    def get_all_configs(self):
        return self.impl

    @contextmanager
    def temp(self, name, value):
        old_value = self[name]
        self.impl[name] = value
        yield
        self.impl[name] = old_value

    def add(self, name: str, type: type, default: Optional[Union[int, str, bool]] = None, env: Union[str, dict] = None):
        update_params = (type, default, env)
        if name not in self._config_params or self._config_params[name] != update_params:
            if name in self._config_params:
                logging.warning(f"The default configuration parameter {name}({self._config_params[name]}) "
                                f"has been added, but a new {name}({update_params}) has been added repeatedly.")
            self._config_params.update({name: update_params})
            if isinstance(env, str):
                self._env_map_name[('lazyllm_' + env).upper()] = name
            elif isinstance(env, dict):
                for k in env.keys():
                    self._env_map_name[('lazyllm_' + k).upper()] = name
        self._update_impl(name, type, default, env)
        return self

    def _update_impl(self, name: str, type: type, default: Optional[Union[int, str, bool]] = None,
                     env: Union[str, dict] = None):
        self.impl[name] = self.cfgs.pop(name) if name in self.cfgs else default
        if isinstance(env, dict):
            for k, v in env.items():
                if self.getenv(k, bool):
                    self.impl[name] = v
                    break
        elif env:
            self.impl[name] = self.getenv(env, type, self.impl[name])
        if not isinstance(self.impl[name], type): raise TypeError(
            f'Invalid config type for {name}, type is {type}')

    def __getitem__(self, name):
        try:
            if isinstance(name, bytes): name = name.decode('utf-8')
            return self.impl[name]
        except KeyError:
            raise RuntimeError(f'Key `{name}` is not in lazyllm global config')

    def __str__(self):
        return str(self.impl)

    def refresh(self, targets: Union[bytes, str, List[str]] = None) -> None:
        names = targets
        if isinstance(targets, bytes): targets = targets.decode('utf-8')
        if isinstance(targets, str):
            names = targets.lower()
            if names.startswith('lazyllm_'):
                names = names[8:]
            names = [names]
        elif targets is None:
            curr_envs = [key for key in os.environ.keys() if key.startswith('LAZYLLM_')]
            names = list(set([self._env_map_name[key] for key in curr_envs if key in self._env_map_name]))
        assert isinstance(names, list)
        for name in names:
            if name in self.impl: self._update_impl(name, *self._config_params[name])

config = Config().add('mode', Mode, Mode.Normal, dict(DISPLAY=Mode.Display, DEBUG=Mode.Debug)
                ).add('repr_ml', bool, False, 'REPR_USE_ML'
                ).add('repr_show_child', bool, False, 'REPR_SHOW_CHILD'
                ).add('rag_store', str, 'none', 'RAG_STORE'
                ).add('gpu_type', str, 'A100', 'GPU_TYPE'
                ).add('train_target_root', str, os.path.join(os.getcwd(), 'save_ckpt'), 'TRAIN_TARGET_ROOT'
                ).add('infer_log_root', str, os.path.join(os.getcwd(), 'infer_log'), 'INFER_LOG_ROOT'
                ).add('temp_dir', str, os.path.join(os.getcwd(), '.temp'), 'TEMP_DIR'
                ).add('thread_pool_worker_num', int, 16, 'THREAD_POOL_WORKER_NUM'
                )
