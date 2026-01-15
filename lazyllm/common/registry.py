import builtins
import functools
import lazyllm
import re
from .bind import _MetaBind
from ..configs import config
from typing import Optional
from abc import ABCMeta

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
        self._default: Optional[str] = None
        self.name = name.capitalize()
        self.base = base

    def __setitem__(self, key, value):
        key = key.lower()
        assert key != 'default', 'LazyDict do not support key: default'
        if '.' in key:
            grp, key = key.rsplit('.', 1)
            return self[grp].__setitem__(key, value)
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        key = key.lower()
        if '.' in key:
            grp, key = key.split('.', 1)
            return self[grp][key]
        return super().__getitem__(key)

    # default -> self.default
    # key -> Key, keyName, KeyName
    # if self.name ends with 's' or 'es', ignor it
    def _match(self, key: str):
        key = key.lower()
        if key == 'default':
            assert self._default or len(self) > 0, 'No default key set'
            key = self._default or list(self.keys())[0]
        keys = [key, f'{key}{self.name}', f'{key}{self.name.lower()}']
        if self.name.endswith('s'):
            n = 2 if self.name.endswith('es') else 1
            keys.extend([f'{key}{self.name[:-n]}', f'{key}{self.name[:-n].lower()}'])

        for k in set(keys):
            if k in self.keys():
                return k
        raise AttributeError(f'Attr {key} not found in `{self.name}: {self}`, conditates: {keys}')

    def __getattr__(self, key):
        return self[self._match(key)]

    def remove(self, key):
        super(__class__, self).pop(self._match(key))

    def __call__(self, *args, **kwargs):
        assert self._default is not None or len(self.keys()) == 1
        return (self.default if self._default else self[list(self.keys())[0]])(*args, **kwargs)

    def set_default(self, key: str):
        assert isinstance(key, str), 'default key must be str'
        self._default = key.lower()

    def __contains__(self, key):
        try:
            _ = self[key]
            return True
        except KeyError:
            return False


group_template = '''\
class LazyLLM{name}Base(LazyLLMRegisterMetaClass.all_clses[\'{base}\'.lower()].base):
    pass
'''

config.add('use_builtin', bool, False, 'USE_BUILTIN',
           description='Whether to use registry modules in python builtin.')


class LazyLLMRegisterMetaClass(_MetaBind):
    all_clses = LazyDict()

    def __new__(metas, name, bases, attrs):
        new_cls = type.__new__(metas, name, bases, attrs)
        if new_cls.__dict__.get('__lazyllm_registry_disable__'): return new_cls
        if name.startswith('LazyLLM') and name.endswith('Base'):
            ori = new_cls.__dict__.get('__lazyllm_registry_key__', re.match('(LazyLLM)(.*)(Base)', name)[2])
            group = ori.lower()
            ori_group = getattr(new_cls, '_lazy_llm_group', '')
            new_cls._lazy_llm_group = f'{ori_group}.{group}'.strip('.')
            ld = LazyDict(group, new_cls)
            if new_cls._lazy_llm_group == group:
                for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                    assert not (hasattr(m, group) and hasattr(m, ori)), f'group name \'{ori}\' cannot be used'
                for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                    setattr(m, group, ld)
                    setattr(m, ori, ld)
            LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group] = ld
            if (f := getattr(new_cls, '__lazyllm_after_registry_hook__', None)):
                f(ori_group, group, isleaf=False)
        elif hasattr(new_cls, '_lazy_llm_group'):
            group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
            name = new_cls.__dict__.get('__lazyllm_registry_key__', name)
            assert name not in group, f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}'
            group[name] = new_cls
            if (f := getattr(new_cls, '__lazyllm_after_registry_hook__', None)):
                f(new_cls._lazy_llm_group, name, isleaf=True)
        return new_cls

    @classmethod
    def _handle_disable_class(cls, new_cls, bases):
        parent_path = cls._get_registry_parent_path(new_cls, bases)
        new_cls._lazy_llm_group = parent_path

    @classmethod
    def _handle_base_class(cls, new_cls, name: str):
        ori = re.match('(LazyLLM)(.*)(Base)', name.split('.')[-1])[2]
        group = ori.lower()

        new_cls._lazy_llm_group = f'{getattr(new_cls, "_lazy_llm_group", "")}.{group}'.strip('.')
        ld = LazyDict(group, new_cls)

        if new_cls._lazy_llm_group == group:
            for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                assert not (hasattr(m, group) and hasattr(m, ori)), f'group name \'{ori}\' cannot be used'
            for m in (builtins, lazyllm) if config['use_builtin'] else (lazyllm,):
                setattr(m, group, ld)
                setattr(m, ori, ld)

        cls.all_clses[new_cls._lazy_llm_group] = ld

    @classmethod
    def _handle_impl_class(cls, new_cls, name):
        group = LazyLLMRegisterMetaClass.all_clses[new_cls._lazy_llm_group]
        assert new_cls.__name__ not in group, (
            f'duplicate class \'{name}\' in group {new_cls._lazy_llm_group}')
        key = new_cls.__name__.lower()
        group[key] = new_cls
        return (new_cls._lazy_llm_group, key)

    @classmethod
    def _get_registry_parent_path(cls, new_cls, bases):
        parent_path = next(
            (gp for base in bases
             if isinstance((gp := getattr(base, '_lazy_llm_group', None)), str) and gp in cls.all_clses),
            None
        )
        assert parent_path is not None, f'No registered parent group found for {new_cls.__name__}'
        return parent_path

    @classmethod
    def _get_registry_attr(cls, attrs, registry_attr_name):
        registry_attr = attrs.get(registry_attr_name, None)
        if registry_attr is not None:
            assert isinstance(registry_attr, str) and registry_attr.strip(), \
                f'{registry_attr_name} must be non-empty str'
            registry_attr = registry_attr.strip().lower()
            assert '.' not in registry_attr, \
                f'Invalid {registry_attr_name}="{registry_attr}". Dot is not allowed.'
        return registry_attr

    @classmethod
    def _handle_registry_class(cls, new_cls, name, bases, attrs):
        parent_path = cls._get_registry_parent_path(new_cls, bases)
        parent_group = cls.all_clses[parent_path]

        registry_key = cls._get_registry_attr(attrs, '__lazyllm_registry_key__')
        registry_group = cls._get_registry_attr(attrs, '__lazyllm_registry_group__')

        target_path = parent_path
        target_group = parent_group

        if registry_group is not None:
            target_path = f'{parent_path}.{registry_group}'
            if registry_group in parent_group:
                target_group = parent_group[registry_group]
                assert isinstance(target_group, LazyDict), \
                    f'Group key conflict: "{registry_group}" exists in "{parent_path}" and is not a LazyDict'
            else:
                target_group = LazyDict(registry_group)
                parent_group[registry_group] = target_group

            cls.all_clses[target_path] = target_group

        assert registry_key not in target_group, \
            f'duplicate registry key "{registry_key}" in group "{target_path}"'
        target_group[registry_key] = new_cls
        new_cls._lazy_llm_group = target_path

        return (target_path, registry_key)

    @classmethod
    def _call_on_register_hook(cls, new_cls, group_path: str, registry_key: str | None):
        hook = getattr(new_cls, '__lazyllm_on_register__', None)
        if hook is None:
            return
        if callable(hook):
            hook(group_path=group_path, registry_key=registry_key)


class LazyLLMRegisterMetaABCClass(LazyLLMRegisterMetaClass, ABCMeta): pass


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

def bind_to_instance(func):
    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class Register(object):
    def __init__(self, base, fnames, template: str = reg_template, default_group: Optional[str] = None):
        self.basecls = base
        self.fnames = [fnames] if isinstance(fnames, str) else fnames
        self.template = template
        self._default_group = default_group
        assert len(self.fnames) > 0, 'At least one function should be given for overwrite.'

    def _wrap(self, cls, *, rewrite_func=None):
        cls = cls.__name__ if isinstance(cls, type) else cls
        cls = re.match('(LazyLLM)(.*)(Base)', cls.split('.')[-1])[2] \
            if (cls.startswith('LazyLLM') and cls.endswith('Base')) else cls
        base = _get_base_cls_from_registry(cls.lower())
        assert issubclass(base, self.basecls)
        if rewrite_func is None:
            rewrite_func = base.__reg_overwrite__ if getattr(base, '__reg_overwrite__', None) else self.fnames[0]
        assert rewrite_func in self.fnames, f'Invalid function "{rewrite_func}" provived for rewrite.'

        def impl(func, func_name=None):
            if func_name:
                func_for_wrapper = func  # avoid calling recursively

                @functools.wraps(func)
                def wrapper_func(*args, **kwargs):
                    return func_for_wrapper(*args, **kwargs)

                wrapper_func.__name__ = func_name
                func = wrapper_func
            else:
                func_name = func.__name__
            exec(self.template.format(
                name=func_name + cls.split('.')[-1].capitalize(), base=cls))
            # 'func' cannot be recognized by exec, so we use 'setattr' instead
            f = LazyLLMRegisterMetaClass.all_clses[cls.lower()].__getattr__(func_name)
            f.__name__ = func_name
            setattr(f, rewrite_func, bind_to_instance(func))
            return func
        return impl

    def __call__(self, f, *, rewrite_func=None):
        if not isinstance(f, (str, type)):
            assert self._default_group, 'default_group is not set, please set it by your register decorator'
            return self._wrap(self._default_group)(f)
        return self._wrap(f, rewrite_func=rewrite_func)

    def __getattr__(self, name):
        if name not in self.fnames:
            raise AttributeError(f'class {self.__class__} has no attribute {name}')

        def impl(cls):
            return self(cls, rewrite_func=name)
        return impl

    def new_group(self, group_name):
        exec('class LazyLLM{name}Base(self.basecls):\n    pass\n'.format(name=group_name))
