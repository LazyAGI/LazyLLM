import builtins
import lazyllm
import re

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
    def __init__(self, name='', base=None, *args, **kw):
        super(__class__, self).__init__(*args, **kw)
        self._default = None
        self.name = name.capitalize()
        self.base = base

    def __setitem__(self, key, value):
        assert key != 'default', 'LazyDict do not support key: default'
        if '.' in key:
            grp, key = key.rsplit('.', 1)
            return self[grp].__setitem__(key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        if '.' in key:
            grp, key = key.split('.', 1)
            return self[grp][key]
        return super().__getitem__(key)

    # default -> self.default
    # key -> Key, keyName, KeyName
    # if self.name ends with 's' or 'es', ignor it
    def __getattr__(self, key):
        key = self._default if key == 'default' else key
        keys = [key, f'{key[0].upper()}{key[1:]}', f'{key}{self.name}', f'{key[0].upper()}{key[1:]}{self.name}',
                f'{key}{self.name.lower()}', f'{key[0].upper()}{key[1:]}{self.name.lower()}']
        if self.name.endswith('s'):
            n = 2 if self.name.endswith('es') else 1
            keys.extend([f'{key}{self.name[:-n]}', f'{key[0].upper()}{key[1:]}{self.name[:-n]}'])

        for k in set(keys):
            if k in self.keys():
                return self[k]
        # return super(__class__, self).__getattribute__(key)
        raise AttributeError(f'Attr {key} not found in {self}')

    def __call__(self, *args, **kwargs):
        assert self._default is not None or len(self.keys()) == 1
        return self.default if self._default else self[list(self.keys())[0]](*args, **kwargs)

    def set_default(self, key):
        assert isinstance(key, str), 'default key must be str'
        self._default = key


group_template = '''\
class LazyLLM{name}Base(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''


class LazyLLMRegisterMetaClass(type):
    all_clses = LazyDict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if name.startswith('LazyLLM') and name.endswith('Base'):
            ori = re.match('(LazyLLM)(.*)(Base)', name.split('.')[-1])[2]
            group = ori.lower()
            new_cls._lazy_llm_group = f'{getattr(new_cls, "_lazy_llm_group", "")}.{group}'.strip('.')
            ld = LazyDict(group, new_cls)
            if new_cls._lazy_llm_group == group:
                for m in (builtins, lazyllm):
                    assert not (hasattr(m, group) and hasattr(m, ori)), f'group name \'{ori}\' cannot be used'
                for m in (builtins, lazyllm):
                    setattr(m, group, ld)
                    setattr(m, ori, ld)
            LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group] = ld
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            assert new_cls.__name__ not in group, (
                f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}')
            group[new_cls.__name__] = new_cls
        return new_cls


def _get_base_cls_from_registry(cls_str, *, registry=LazyLLMRegisterMetaClass.all_clses):
    if cls_str == '':
        return registry.base
    group, cls_str = cls_str.split('.', 1) if '.' in cls_str else (cls_str, '')
    if not (registry is LazyLLMRegisterMetaClass.all_clses or group in registry):
        exec(group_template.format(name=group.capitalize(), base=registry.base._lazy_llm_group))
    return _get_base_cls_from_registry(cls_str, registry=registry[group])


reg_template = '''\
class {name}(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''


class Register(object):
    def __init__(self, base, fnames, template=reg_template):
        self.basecls = base
        self.fnames = fnames
        self.template = template
        assert len(self.fnames) > 0, 'At least one function should be given for overwrite.'

    def __call__(self, cls, *, rewrite_func=None):
        cls = cls.__name__ if isinstance(cls, type) else cls
        cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
            if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
        base = _get_base_cls_from_registry(cls.lower())
        assert issubclass(base, self.basecls)
        if rewrite_func is None:
            rewrite_func = base.__reg_overwrite__ if getattr(base, '__reg_overwrite__', None) else self.fnames[0]
        assert rewrite_func in self.fnames, f'Invalid function "{rewrite_func}" provived for rewrite.'

        def impl(func):
            func_name = func.__name__
            exec(self.template.format(
                name=func_name + cls.split('.')[-1].capitalize(), base=cls))
            # 'func' cannot be recognized by exec, so we use 'setattr' instead
            f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)
            f.__name__ = func_name
            setattr(f, rewrite_func, lambda _, *args, **kw: func(*args, **kw))
            return func
        return impl

    def __getattr__(self, name):
        if name not in self.fnames:
            raise AttributeError(f'class {self.__class__} has no attribute {name}')

        def impl(cls):
            return self(cls, rewrite_func=name)
        return impl

    def new_group(self, group_name):
        exec('class LazyLLM{name}Base(self.basecls):\n    pass\n'.format(name=group_name))
