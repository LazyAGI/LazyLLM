import os
from enum import Enum
import json


class Mode(Enum):
    Display = 0,
    Normal = 1,
    Debug = 2,


class Config(object):
    def __init__(self, prefix='LAZYLLM', home='~/.lazyllm/'):
        self.prefix = prefix
        self.impl, self.cfgs = dict(), dict()
        self.add('home', str, home, 'HOME')
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

    def add(self, name, type, default=None, env=None):
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
        return self

    def __getitem__(self, name):
        return self.impl[name]

    def __str__(self):
        return str(self.impl)


config = Config().add('mode', Mode, Mode.Normal, dict(DISPLAY=Mode.Display, DEBUG=Mode.Debug)
                ).add('repr_ml', bool, False, 'REPR_USE_ML'
                ).add('rag_store', str, 'none', 'RAG_STORE'
                ).add('redis_url', str, 'none', 'REDIS_URL'
                ).add('gpu_type', str, 'A100', 'GPU_TYPE'
                )
