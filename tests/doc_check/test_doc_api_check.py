import pytest # noqa E401
import re
import inspect
import lazyllm
from typing import Callable
import warnings


def class_should_check(cls, module):
    if not cls.__name__[0].isupper() or cls.__module__ != module.__name__:
        return False
    if cls.__module__ != module.__name__:
        return False
    all_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    custom_methods = [name for name, func in all_methods if not name.startswith('_')]
    return len(custom_methods) > 0


def get_sub_classes(module):
    clsmembers = inspect.getmembers(module, inspect.isclass)
    classes = set([ele[1] for ele in clsmembers if class_should_check(ele[1], module)])
    for name, sub_module in inspect.getmembers(module, inspect.ismodule):
        if sub_module.__name__.startswith(module.__name__):
            classes.update(get_sub_classes(sub_module))
    return classes


def is_method_overridden(cls, method: Callable):
    method_name = method.__name__
    for base in cls.__bases__:
        if hasattr(base, method_name):
            base_method = getattr(base, method_name)
            current_method = getattr(cls, method_name)
            if current_method != base_method:
                return True
    return False


def do_check_method(cls, func: Callable):
    # As type is always missing in code signature and default value is not universal,
    # Also Keyword argument is not universal. So we just check args parameter name
    arg_spec = inspect.getfullargspec(func)
    real_parms = arg_spec.args + arg_spec.kwonlyargs
    real_vars = [arg_spec.varargs, arg_spec.varkw]
    if real_parms[0] in ['self', 'cls']:
        real_parms = real_parms[1:]
    real_parms = set(real_parms + real_vars)
    if func.__name__ == '__init__':
        doc = cls.__doc__
    else:
        doc = func.__doc__
    if doc is not None:
        seg_pattern = r"Args:\s*(.*?)\n\s*\n"
        match = re.search(seg_pattern, doc, re.DOTALL)
        doc_parms = []
        if match:
            args_pattern = r"^\s*(\w+)\s*(?:\(|:)"
            doc_parms = re.findall(args_pattern, match.group(1), re.MULTILINE)
        for doc_param in doc_parms:
            if doc_param in real_parms:
                continue
            assert doc_param in real_parms, f"{doc_param} no found in real params: {real_parms}"
    else:
        if len(real_parms) > 0:
            warnings.warn(f"doc is empty, real params: {real_parms}", UserWarning)


def create_test_function(cls, func):
    if func.__name__ == "__init__":
        dynamic_func_name = f"test_{cls.__name__}"
    else:
        dynamic_func_name = f"test_{cls.__name__}_{func.__name__}"
    while dynamic_func_name in global_func_names:
        dynamic_func_name = dynamic_func_name + "_"
    global_func_names.add(dynamic_func_name)
    cls_path = f"{cls.__module__}.{cls.__qualname__}"
    func_path = f"{cls_path}.{func.__name__}"
    xfail_decorator = "@pytest.mark.xfail"
    code = f"{xfail_decorator}\ndef {dynamic_func_name}():\n    do_check_method({cls_path}, {func_path})"
    exec(code, globals())


def gen_check_cls_and_funtions():
    all_classes = get_sub_classes(lazyllm)
    for cls in all_classes:
        all_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        custom_methods = [func for name, func in all_methods if not name.startswith('_') or name == '__init__']
        overridden_methods = [func for func in custom_methods if is_method_overridden(cls, func)]
        for overridden_method in overridden_methods:
            create_test_function(cls, overridden_method)


global_func_names = set()
gen_check_cls_and_funtions()
