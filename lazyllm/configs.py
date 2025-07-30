import os
from enum import Enum
import json
from typing import List, Union
from contextlib import contextmanager
import logging


class Mode(Enum):
    """An enumeration."""
    Display = 0,
    Normal = 1,
    Debug = 2,


class Config(object):
    """Config is a configuration class provided by LazyLLM, which loads configurations of LazyLLM framework from config files,
environment variables, or specify them explicitly. it can export all configuration items as well.
The Config module automatically generates an object named 'config' containing all configurations.
"""
    def __init__(self, prefix='LAZYLLM', home=os.path.join(os.path.expanduser('~'), '.lazyllm')):
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
        """Check if any configuration items in the config.json file that is not loaded by the add method.

Args:
    None.
"""
        assert len(self.cfgs) == 0, f'Invalid cfgs ({"".join(self.cfgs.keys())}) are given in {self.cgf_path}'
        return self

    def getenv(self, name, type, default=None):
        """Get value of LazyLLM-related environment variables.

Args:
    name (str): The name of the environment variable （without the prefix）, case-insensitive. The function obtains value
    from environment variable by concatenating the prefix and this name, with all uppercase letters.
    type (type): Specifies the type of the configuration, for example, str. For boolean types, the function will
    convert inputs ‘TRUE’, ‘True’, 1, ‘ON’, and ‘1’ to True.
    default (optional): If the value of the environment variable cannot be obtained, this value is returned.
"""
        r = os.getenv(f'{self.prefix}_{name.upper()}', default)
        if type == bool:
            return r in (True, 'TRUE', 'True', 1, 'ON', '1')
        return type(r)

    @staticmethod
    def get_config(cfg):
        return cfg

    def get_all_configs(self):
        """Get all configurations from the config.

Args:
    None.



Examples:
    >>> import lazyllm
    >>> from lazyllm.configs import config
    >>> config['launcher']
    'empty'
    >>> config.get_all_configs()
    {'home': '~/.lazyllm/', 'mode': <Mode.Normal: (1,)>, 'repr_ml': False, 'rag_store': 'None', 'redis_url': 'None', ...}
    """
        return self.impl

    @contextmanager
    def temp(self, name, value):
        old_value = self[name]
        self.impl[name] = value
        yield
        self.impl[name] = old_value

    def add(self, name, type, default=None, env=None):
        """Loads value into LazyLLM configuration item. The function first attempts to find the value with the given name from the
dict loaded from config.json. If found, it removes the key from the dict and saves the value to the config.
If 'env' is a string, the function calls getenv to look for the corresponding LazyLLM environment variable, and if
it's found, writes it to the config. If 'env' is a dictionary, the function attempts to call getenv to find the
environment variables corresponding to the keys in the dict and convert them to boolean type.
If the converted boolean value is True, the value corresponding to the current key in the dict is written to the config.

Args:
    name (str): The name of the configuration item
    type (type): The type of the configuration
    default (optional): The default value of the configuration if no value can be obtained
    env (optional): The name of the environment variable without the prefix, or a dictionary where the keys are the
    names of the environment variables(without the prefix), and the values are what to be added to the configuration.
"""
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

    def _update_impl(self, name, type, default=None, env=None):
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
            return self.impl[name]
        except KeyError:
            raise RuntimeError(f'Key {name} is not in lazyllm global config')

    def __str__(self):
        return str(self.impl)

    def refresh(self, targets: Union[str, List[str]] = None) -> None:
        names = targets
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
            self._update_impl(name, *self._config_params[name])

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
