import pytest # noqa E401
import re
import inspect
import lazyllm
from typing import Callable
import warnings
import dataclasses
import enum


def class_should_check(cls, module):
    """Check if a class should be included in documentation verification."""
    # Skip dataclass and enum classes
    if dataclasses.is_dataclass(cls) or issubclass(cls, enum.Enum):
        return False

    if not cls.__name__[0].isupper() or cls.__module__ != module.__name__:
        return False

    all_methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    custom_methods = [name for name, func in all_methods if not name.startswith('_')]
    return len(custom_methods) > 0


def get_sub_classes(module):
    """Get all valid subclasses from a module recursively."""
    clsmembers = inspect.getmembers(module, inspect.isclass)
    classes = set([ele[1] for ele in clsmembers if class_should_check(ele[1], module)])
    for _, sub_module in inspect.getmembers(module, inspect.ismodule):
        if sub_module.__name__.startswith(module.__name__):
            classes.update(get_sub_classes(sub_module))
    return classes


def is_method_overridden(cls, method: Callable):
    """Check if a method is overridden from its parent class."""
    method_name = method.__name__
    for base in cls.__bases__:
        if hasattr(base, method_name):
            base_method = getattr(base, method_name)
            current_method = getattr(cls, method_name)
            if current_method != base_method:
                return True
    return False


def is_doc_directly_written(cls, func: Callable) -> bool:
    """Check if documentation is written directly in the function/class."""
    try:
        if func.__name__ == '__init__':
            source = inspect.getsource(cls)
        else:
            source = inspect.getsource(func)

        docstring_pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        matches = re.findall(docstring_pattern, source)
        return len(matches) > 0
    except (TypeError, OSError):
        return False


def get_doc_from_language(cls, func: Callable, language: str = 'ENGLISH'):
    """Get documentation based on language configuration."""
    # First check if doc is written directly in the function
    if is_doc_directly_written(cls, func):
        warnings.warn(
            f"Documentation for {cls.__name__}.{func.__name__} is written directly in the "
            f"function/class. Please use add_{language.lower()}_doc instead.",
            UserWarning, stacklevel=2
        )
        return None

    # Get documentation using temporary language configuration
    with lazyllm.config.temp('language', language):
        if func.__name__ == '__init__':
            doc = cls.__doc__
        else:
            doc = func.__doc__
        return doc


def parse_google_style_args(doc: str) -> set:
    """Parse Args section from Google style docstring and return parameter names."""
    args_pattern = r"Args:\s*(.*?)(?:\n\s*(?:Returns|Raises|$))"
    args_match = re.search(args_pattern, doc, re.DOTALL)
    if not args_match:
        return set()

    args_section = args_match.group(1)
    param_pattern = r"^\s*(\w+)\s*(?:\([^)]+\))?\s*:"
    params = set()

    for line in args_section.split('\n'):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        match = re.search(param_pattern, line)
        if match:
            param_name = match.group(1)
            params.add(param_name)

    return params


def do_check_method(cls, func: Callable):
    """Check if method's parameter documentation matches actual parameters."""
    # Get actual function parameters
    arg_spec = inspect.getfullargspec(func)
    real_params = arg_spec.args + (arg_spec.kwonlyargs or [])
    if real_params and real_params[0] in ['self', 'cls']:
        real_params = real_params[1:]
    real_params = set(real_params)

    # Check if documentation is written directly in code
    if is_doc_directly_written(cls, func):
        warnings.warn(
            f"Documentation for {cls.__name__}.{func.__name__} is written directly in the "
            f"function/class. Please use add_english_doc and add_chinese_doc instead.",
            UserWarning, stacklevel=2
        )
        return

    # Check documentation in both English and Chinese environments
    with lazyllm.config.temp('language', 'ENGLISH'):
        eng_doc = func.__doc__ if func.__name__ != '__init__' else cls.__doc__
        if eng_doc:
            check_doc_params(eng_doc, real_params, 'ENGLISH')

    with lazyllm.config.temp('language', 'CHINESE'):
        cn_doc = func.__doc__ if func.__name__ != '__init__' else cls.__doc__
        if cn_doc:
            check_doc_params(cn_doc, real_params, 'CHINESE')


def check_doc_params(doc: str, real_params: set, language: str):
    """Check if documentation parameters match actual parameters"""
    # Check documentation format
    if not re.search(r"Args:", doc):
        raise ValueError(f"[{language}] Missing 'Args:' section in docstring")

    # Parse parameters from documentation
    doc_params = parse_google_style_args(doc)

    # Check parameter completeness
    missing_params = real_params - doc_params
    extra_params = doc_params - real_params

    if missing_params:
        raise ValueError(f"[{language}] Parameters missing in docstring: {', '.join(missing_params)}")
    if extra_params:
        raise ValueError(f"[{language}] Extra parameters in docstring: {', '.join(extra_params)}")


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

    code = f"""def {dynamic_func_name}():
    print(f'\\nChecking {cls.__name__}.{func.__name__}')
    do_check_method({cls_path}, {func_path})
"""
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
