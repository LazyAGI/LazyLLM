import re
import builtins

import lazyllm


# Special Dict for lazy programmer. Suppose we have a LazyDict as follows：
#    >>> ld = LazyDict(name='ld', ALd=int)
# 1. Use dot instead of ['str']
#    >>> ld.ALd
# 2. Support lowercase first character to make the sentence more like a function
#    >>> ld.aLd
# 3. Supports direct calls to dict when there is only one element
#    >>> ld()
# 4. Support dynamic default key
#    >>> ld.set_default('ALd')
#    >>> ld.default
# 5. allowed to omit the group name if the group name appears in the name
#    >>> ld.a
class LazyDict(dict):
    def __init__(self, name, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._default = None
        self.name = name.capitalize()

    def __setitem__(self, key, value):
        assert key != 'default', 'LazyDict do not support key: default'
        return super().__setitem__(key, value)
    
    # default -> self.default
    # key -> Key, keyName, KeyName
    def __getattr__(self, key):
        key = self._default if key == 'default' else key
        for k in (key, f'{key[0].upper()}{key[1:]}', f'{key}{self.name}', f'{key[0].upper()}{key[1:]}{self.name}'):
            if k in self.keys():
                return self[k]
        return super(__class__, self).__getattribute__(key)

    def __call__(self, *args, **kwargs):
        assert self._default is not None or len(self.keys()) == 1
        return self.default if self._default else self[list(self.keys())[0]](*args, **kwargs)

    def set_default(self, key):
        assert isinstance(key, str), 'default key must be str'
        self._default = key
        

class LazyLLMRegisterMetaClass(type):
    all_clses = dict()
    all_groups = dict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if name.startswith('LazyLLM') and name.endswith('Base'):
            group = re.match('(LazyLLM)(.*)(Base)', name.split('.')[-1])[2].lower()
            assert not hasattr(new_cls, '_lazy_llm_group')
            new_cls._lazy_llm_group = group

            LazyLLMRegisterMetaClass.all_clses.update({group:LazyDict(group)})
            LazyLLMRegisterMetaClass.all_groups.update({group:new_cls})

            assert not hasattr(builtins, group), f'group name \'{group}\' cannot be used'
            setattr(builtins, group, LazyLLMRegisterMetaClass.all_clses[group])
            setattr(lazyllm, group, LazyLLMRegisterMetaClass.all_clses[group])
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            assert new_cls.__name__ not in group, (
                f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}')
            group[new_cls.__name__] = new_cls
        return new_cls