import os
from enum import Enum
import json
from typing import List, Union


class Mode(Enum):
    Display = 0,
    Normal = 1,
    Debug = 2,


class Config(object):
    def __init__(self, prefix='LAZYLLM', home='~/.lazyllm/'):
        self._default_add = dict()
        self.prefix = prefix
        self.impl, self.cfgs = dict(), dict()
        self.add('home', str, home, 'HOME')
        os.system(f'mkdir -p {home}')
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
        update_params = (type, default, env)
        if name not in self._default_add or self._default_add[name] != update_params:
            if name in self._default_add:
                print(f"Warning: The default configuration parameter {name}({self._default_add[name]}) has been added, "
                      f"but a new {name}({update_params}) has been added repeatedly.")
            self._default_add.update({name: update_params})
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

    def refresh(self, targets: Union[str, List[str]] = None) -> None:
        names = targets
        if isinstance(targets, str):
            names = targets.lower()
            if names.startswith('lazyllm_'):
                names = names[8:]
            names = [names]
        elif targets is None:
            names = list(self.impl.keys())
        assert isinstance(names, list)
        for name in names:
            self.add(name, *self._default_add[name])

config = Config().add('mode', Mode, Mode.Normal, dict(DISPLAY=Mode.Display, DEBUG=Mode.Debug)
                ).add('repr_ml', bool, False, 'REPR_USE_ML'
                ).add('rag_store', str, 'none', 'RAG_STORE'
                ).add('gpu_type', str, 'A100', 'GPU_TYPE'
                )
