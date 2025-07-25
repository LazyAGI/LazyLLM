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
    """\
Config是LazyLLM提供的配置类，可以支持通过加载配置文件、设置环境变量、编码写入默认值等方式设置LazyLLM框架的相关配置，以及导出当前所有的配置项和对应的值。
Config模块自动生成一个config对象，其中包含所有的配置。
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
        """\
用于检查config.json配置文件中是否还有没有通过add方法载入的配置项
"""
        assert len(self.cfgs) == 0, f'Invalid cfgs ({"".join(self.cfgs.keys())}) are given in {self.cgf_path}'
        return self

    def getenv(self, name, type, default=None):
        """\
用于检查config.json配置文件中是否还有没有通过add方法载入的配置项
Config.getenv(name, type, default): -> str\n
获取LazyLLM相关环境变量的值

Args:
    name (str): 不包含前缀的环境变量名字，不区分大小写。函数将从拼接了前缀和此名字的全大写的环境变量中获取对应的值。
    type (type): 指定该配置的类型，例如str。对于bool型，函数会将'TRUE', 'True', 1, 'ON', '1'这5种输入转换为True。
    default (可选): 若无法获取到环境变量的值，将返回此变量。

"""
        r = os.getenv(f'{self.prefix}_{name.upper()}', default)
        if type == bool:
            return r in (True, 'TRUE', 'True', 1, 'ON', '1')
        return type(r)

    @staticmethod
    def get_config(cfg):
        return cfg

    def get_all_configs(self):
        """\
获取config中所有的配置



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
        """\
将值加载至LazyLLM的配置项中。函数首先尝试从加载自config.json的字典中查找名字为name的值。若找到则从该字典中删去名为name的键。并将对应的值写入config。
若env是一个字符串，函数会调用getenv寻找env对应的LazyLLM环境变量，若找到则写入config。如果env为一个字典，函数将尝试调用getenv寻找字典中的key对于的环境变量并转换为bool型。
若转换完成的bool值是True，则将字典中当前的key对应的值写入config。

Args:
    name (str): 配置项名称
    type (type): 该配置的类型
    default (可选): 若无法获取到任何值，该配置的默认值
    env (可选): 不包含前缀的环境变量名称，或者一个字典，其中的key是不包含前缀的环境变量名称，value是要加入配置的值。

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
