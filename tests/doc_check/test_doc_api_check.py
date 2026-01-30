import pytest
import inspect
import os
import sys
import re
import dataclasses
import enum
import logging
from typing import Callable

os.environ['LAZYLLM_INIT_DOC'] = 'True'
current_dir = os.path.dirname(os.path.abspath(__file__))
lazyllm_root = os.path.dirname(os.path.dirname(current_dir))

if lazyllm_root not in sys.path:
    sys.path.insert(0, lazyllm_root)

DOCS_SCRIPTS_PATH = os.path.join(lazyllm_root, 'docs', 'scripts')
if DOCS_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, DOCS_SCRIPTS_PATH)

if 'lazyllm' in sys.modules:
    del sys.modules['lazyllm']

try:
    import lazyllm
    from lazynote.manager import SimpleManager
except ImportError as e:
    logging.error(f'Error import lazyllm and lazynote: {e}')
    SimpleManager = None

def generate_docs_for_module():
    if SimpleManager is None:
        return False

    try:
        os.environ['LAZYLLM_LANGUAGE'] = 'CHINESE'

        skip_list = [
            'lazyllm.components.deploy.relay.server',
            'lazyllm.components.deploy.relay.base',
            'lazyllm.components.finetune.easyllm',
            'lazyllm.tools.rag.component.bm25_retriever',
            'lazyllm.tools.memory',
            'lazyllm.tools.infer_service',
            'lazyllm.tools.train_service',
        ]

        manager = SimpleManager(pattern='clear', skip_on_error=True)
        manager.traverse(lazyllm, skip_modules=skip_list)

        manager = SimpleManager(pattern='fill', skip_on_error=True)
        manager.traverse(lazyllm, skip_modules=skip_list)

        return True

    except Exception:
        return False

def class_should_check(cls, module):
    if dataclasses.is_dataclass(cls) or issubclass(cls, enum.Enum):
        return False

    if not cls.__name__[0].isupper() or cls.__module__ != module.__name__:
        return False

    all_methods = []
    for name, obj in inspect.getmembers(cls):
        if (inspect.isfunction(obj) or isinstance(obj, classmethod) or isinstance(obj, staticmethod)):
            all_methods.append((name, obj))

    custom_methods = [name for name, obj in all_methods
                      if not name.startswith('_') or name == '__init__']
    return len(custom_methods) > 0

def get_sub_classes(module):
    try:
        clsmembers = inspect.getmembers(module, inspect.isclass)
    except Exception:
        return set()

    classes = set([ele[1] for ele in clsmembers if class_should_check(ele[1], module)])

    for _, sub_module in inspect.getmembers(module, inspect.ismodule):
        if sub_module.__name__.startswith(module.__name__):
            if 'thirdparty' in sub_module.__name__ or 'ChatTTS' in sub_module.__name__:
                continue
            try:
                classes.update(get_sub_classes(sub_module))
            except Exception:
                continue
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

def find_highest_ancestor_with_method(cls, method_name: str):
    highest = None
    for base in cls.__bases__:
        if base is object:
            continue
        if hasattr(base, method_name):
            highest = base
        ancestor = find_highest_ancestor_with_method(base, method_name)
        if ancestor is not None:
            highest = ancestor
    return highest

def check_method_has_doc(cls, func: Callable) -> tuple[bool, bool]:
    has_chinese_doc = False
    has_english_doc = False

    try:
        with lazyllm.config.temp('language', 'CHINESE'):
            if func.__name__ == '__init__':
                doc = cls.__doc__
            else:
                doc = func.__doc__
            has_chinese_doc = bool(doc and doc.strip())
    except Exception:
        pass

    try:
        with lazyllm.config.temp('language', 'ENGLISH'):
            if func.__name__ == '__init__':
                doc = cls.__doc__
            else:
                doc = func.__doc__
            has_english_doc = bool(doc and doc.strip())
    except Exception:
        pass

    return has_chinese_doc, has_english_doc

def do_check_method(cls, func: Callable):
    try:
        if func.__name__ != '__init__':
            if is_method_overridden(cls, func):
                highest_ancestor = find_highest_ancestor_with_method(cls, func.__name__)
                if highest_ancestor:
                    ancestor_has_chinese, ancestor_has_english = check_method_has_doc(
                        highest_ancestor, getattr(highest_ancestor, func.__name__)
                    )

                    if ancestor_has_chinese or ancestor_has_english:
                        return

        has_chinese_doc, has_english_doc = check_method_has_doc(cls, func)

        if not has_chinese_doc and not has_english_doc:
            error_msg = f'No documentation found for {cls.__name__}.{func.__name__} after doc generation'
            raise ValueError(error_msg)

    except Exception:
        raise

def get_defined_methods(cls):
    try:
        source = inspect.getsource(cls)
        method_pattern = r'(?:@\w+\s+)*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return set(re.findall(method_pattern, source))
    except (TypeError, OSError):
        return set()

_docs_generated = False

@pytest.fixture(scope='session', autouse=True)
def generate_docs():
    global _docs_generated
    if not _docs_generated:
        generate_docs_for_module()
        _docs_generated = True
    return _docs_generated

def create_test_function(cls, func):
    dynamic_func_name = f'test_{cls.__name__}_{func.__name__}'

    while dynamic_func_name in global_func_names:
        dynamic_func_name = dynamic_func_name + '_'
    global_func_names.add(dynamic_func_name)
    cls_path = f'{cls.__module__}.{cls.__qualname__}'
    func_path = f'{cls_path}.{func.__name__}'

    code = f'''def {dynamic_func_name}():
    do_check_method({cls_path}, {func_path})
'''
    exec(code, globals())

def gen_check_cls_and_funtions():  # noqa: C901
    all_classes = get_sub_classes(lazyllm)
    for cls in all_classes:
        defined_methods = get_defined_methods(cls)

        all_methods = []
        for name, obj in cls.__dict__.items():
            if name not in defined_methods:
                continue

            if inspect.isfunction(obj):
                try:
                    if obj.__module__ != cls.__module__:
                        continue
                except (AttributeError, TypeError):
                    continue

            if isinstance(obj, classmethod):
                all_methods.append((name, obj.__func__))
            elif isinstance(obj, staticmethod):
                all_methods.append((name, obj.__func__))
            elif inspect.isfunction(obj):
                all_methods.append((name, obj))

        custom_methods = [
            func for name, func in all_methods
            if not name.startswith('_') or name == '__init__'
        ]

        for method in custom_methods:
            create_test_function(cls, method)

global_func_names = set()
gen_check_cls_and_funtions()

def run_all_tests():
    generate_docs_for_module()
    test_functions = [name for name in globals() if name.startswith('test_') and callable(globals()[name])]

    for test_name in test_functions:
        try:
            globals()[test_name]()
        except Exception:
            pass

if __name__ == '__main__':
    run_all_tests()
