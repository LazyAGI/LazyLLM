import pytest
import re
import inspect
import os
from typing import Callable
import dataclasses
import enum
import lazyllm


def should_check_class(cls, module):
    """Check if a class should be included in documentation verification."""
    if dataclasses.is_dataclass(cls) or issubclass(cls, enum.Enum):
        return False

    if not cls.__name__[0].isupper() or cls.__module__ != module.__name__:
        return False

    all_methods = []
    for name, obj in inspect.getmembers(cls):
        if (inspect.isfunction(obj)
                or isinstance(obj, classmethod)
                or isinstance(obj, staticmethod)):
            all_methods.append((name, obj))

    custom_methods = [name for name, obj in all_methods
                      if not name.startswith('_') or name == '__init__']
    return len(custom_methods) > 0


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


def get_sub_classes(module):
    """Get all valid subclasses from a module recursively."""
    try:
        clsmembers = inspect.getmembers(module, inspect.isclass)
    except Exception as e:
        print(f"Warning: Failed to inspect module {module.__name__}: {e}")
        return set()

    classes = set([ele[1] for ele in clsmembers if should_check_class(ele[1], module)])

    for name, sub_module in inspect.getmembers(module, inspect.ismodule):
        if sub_module.__name__.startswith(module.__name__):
            if 'thirdparty' in sub_module.__name__ or 'ChatTTS' in sub_module.__name__:
                print(f"Skipping problematic module: {sub_module.__name__}")
                continue
            try:
                classes.update(get_sub_classes(sub_module))
            except Exception as e:
                print(f"Warning: Failed to process submodule {sub_module.__name__}: {e}")
                continue
    return classes


def _get_module_parts(cls):
    """Extract module information for documentation search."""
    module_name = cls.__module__
    class_name = cls.__name__

    if module_name.startswith('lazyllm.'):
        relative_module = module_name[8:]  # Remove 'lazyllm.' prefix
        module_parts = relative_module.split('.')
        top_module = module_parts[0] if module_parts else ''
    else:
        top_module = ''

    return class_name, top_module


def _build_possible_doc_names(func, class_name, top_module):
    """Construct possible documentation names for a function."""
    possible_doc_names = []

    if func.__name__ == '__init__':
        # For __init__ method, try following formats:
        # 1. Full format: module.class (e.g. deploy.Lightllm)
        # 2. Simple format: class (e.g. FlowBase)
        if top_module:
            possible_doc_names.append(f"{top_module}.{class_name}")
        possible_doc_names.append(class_name)
    else:
        # For regular methods, try following formats:
        # 1. Full format: module.class.method (e.g. rag.DocManager.document)
        # 2. Simple format: class.method (e.g. FlowBase.is_root)
        if top_module:
            possible_doc_names.append(f"{top_module}.{class_name}.{func.__name__}")
        possible_doc_names.append(f"{class_name}.{func.__name__}")

    return possible_doc_names


def _search_doc_patterns_in_file(file_path, possible_doc_names):
    """Search for documentation patterns in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Match each possible documentation name
            for doc_name in possible_doc_names:
                # Flexible matching pattern supporting both single and double quotes
                patterns = [
                    f"add_chinese_doc\\(['\"]({re.escape(doc_name)})['\"]\\s*,",
                    f"add_english_doc\\(['\"]({re.escape(doc_name)})['\"]\\s*,"
                ]

                for pattern in patterns:
                    if re.search(pattern, content):
                        print(f"Found documentation for '{doc_name}' in {file_path}")
                        return True

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return False


def find_doc_in_docs_dir(cls, func: Callable) -> bool:
    """Check if documentation is added through add_*_doc functions."""
    try:
        docs_dir = os.path.join(os.path.dirname(lazyllm.__file__), 'docs')
        if not os.path.exists(docs_dir):
            return False

        class_name, top_module = _get_module_parts(cls)
        possible_doc_names = _build_possible_doc_names(func, class_name, top_module)

        # Search through all Python files in the docs directory
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = os.path.join(root, file)
                if _search_doc_patterns_in_file(file_path, possible_doc_names):
                    return True

        print(f"No documentation found for any of: {possible_doc_names}")
        return False
    except Exception as e:
        print(f"Error checking docs directory: {e}")
        return False


def parse_google_style_args(doc: str) -> set:
    """Parse Args section from Google style docstring and return parameter names."""
    args_pattern = r"Args:\s*(.*?)(?:\n\s*(?:Returns|Raises|Examples|Note|$))"
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


def find_highest_ancestor_with_method(cls, method_name: str):
    """Find the highest ancestor class that has the given method."""
    highest = None
    for base in cls.__bases__:
        if base is object:  # Skip the base object class
            continue
        # Check if current base class has the method
        if hasattr(base, method_name):
            highest = base
        # Check higher ancestors
        ancestor = find_highest_ancestor_with_method(base, method_name)
        if ancestor is not None:
            highest = ancestor
    return highest


def do_check_method(cls, func: Callable):  # noqa: C901
    """Check if method's parameter documentation matches actual parameters."""
    try:
        # For __init__ method, always check (never skip) because each class needs its own documentation
        if func.__name__ != '__init__':
            # For non-__init__ methods, check if ancestor has documentation
            if is_method_overridden(cls, func):
                highest_ancestor = find_highest_ancestor_with_method(cls, func.__name__)
                if highest_ancestor:
                    # Check if the ancestor class has documentation for this method through add_*_doc
                    ancestor_method = getattr(highest_ancestor, func.__name__)
                    ancestor_has_doc = find_doc_in_docs_dir(highest_ancestor, ancestor_method)

                    # If ancestor has documentation through add_*_doc, skip checking this class
                    if ancestor_has_doc:
                        ancestor_name = highest_ancestor.__name__
                        print(f"Skipping {cls.__name__}.{func.__name__} as ancestor class "
                              f"{ancestor_name} has documentation through add_*_doc")
                        return

        # Check if documentation is added through add_*_doc
        has_add_doc = find_doc_in_docs_dir(cls, func)

        if not has_add_doc:
            # No documentation added through add_*_doc
            error_msg = f"Missing documentation through add_*_doc for {cls.__name__}.{func.__name__}"
            raise ValueError(error_msg)

        # If add_*_doc documentation exists, check parameter matching
        # Get actual function parameters
        arg_spec = inspect.getfullargspec(func)
        real_params = arg_spec.args + (arg_spec.kwonlyargs or [])
        if real_params and real_params[0] in ['self', 'cls']:
            real_params = real_params[1:]
        real_params = set(real_params)

        # Check if documentation exists in at least one language and check parameters
        has_english_doc = False
        has_chinese_doc = False

        try:
            with lazyllm.config.temp('language', 'ENGLISH'):
                eng_doc = func.__doc__ if func.__name__ != '__init__' else cls.__doc__
                if eng_doc and eng_doc.strip():
                    has_english_doc = True
                    parse_style_check(eng_doc, real_params, 'ENGLISH')
        except ValueError:
            # Parameter related error
            raise

        try:
            with lazyllm.config.temp('language', 'CHINESE'):
                cn_doc = func.__doc__ if func.__name__ != '__init__' else cls.__doc__
                if cn_doc and cn_doc.strip():
                    has_chinese_doc = True
                    parse_style_check(cn_doc, real_params, 'CHINESE')
        except ValueError:
            # Parameter related error
            raise

        # If documentation is added through add_*_doc but not accessible
        if not has_english_doc and not has_chinese_doc:
            error_msg = f"Documentation added through add_*_doc but not accessible for {cls.__name__}.{func.__name__}"
            raise ValueError(error_msg)

    except Exception:
        raise


# Initialize the set to track reported errors
do_check_method.reported_errors = set()


def parse_style_check(doc: str, real_params: set, language: str):
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
    """Create a test function for checking method documentation."""
    # Always include the method name in the test function name
    dynamic_func_name = f"test_{cls.__name__}_{func.__name__}"

    while dynamic_func_name in global_func_names:
        dynamic_func_name = dynamic_func_name + "_"
    global_func_names.add(dynamic_func_name)
    cls_path = f"{cls.__module__}.{cls.__qualname__}"
    func_path = f"{cls_path}.{func.__name__}"

    code = f"""def {dynamic_func_name}():
    print(f'\\nChecking {cls.__name__}.{func.__name__}')
    try:
        do_check_method({cls_path}, {func_path})
    except Exception as e:
        # Display failure message and re-raise for pytest
        print(f'FAILED: {cls.__name__}.{func.__name__} - {{str(e)}}')
        raise
"""
    exec(code, globals())


def get_defined_methods(cls):
    """Get methods that are actually defined in the class.

    This function only returns methods that are defined in the class's source code,
    excluding methods that are dynamically added or inherited from parent classes.
    """
    try:
        source = inspect.getsource(cls)
        # Improved regex that can match method definitions with decorators
        method_pattern = r'(?:@\w+\s+)*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return set(re.findall(method_pattern, source))
    except (TypeError, OSError):
        return set()


def gen_check_cls_and_funtions():
    """Generate test functions for all classes and their methods."""
    all_classes = get_sub_classes(lazyllm)
    for cls in all_classes:
        # Only care about methods that are truly defined in current class source code
        defined_methods = get_defined_methods(cls)

        all_methods = []
        # Iterate through cls.__dict__ directly to avoid descriptor unpacking
        for name, obj in cls.__dict__.items():
            if name not in defined_methods:          # Filter out non-source-defined methods
                continue

            if isinstance(obj, classmethod):
                # The actual function for classmethod/staticmethod is in __func__
                all_methods.append((name, obj.__func__))
            elif isinstance(obj, staticmethod):
                all_methods.append((name, obj.__func__))
            elif inspect.isfunction(obj):
                all_methods.append((name, obj))

        # Only check non-private methods, with __init__ as exception
        custom_methods = [
            func for name, func in all_methods
            if not name.startswith('_') or name == '__init__'
        ]
        print(f"Custom methods to check: {[m.__name__ for m in custom_methods]}")

        for method in custom_methods:
            create_test_function(cls, method)


global_func_names = set()
gen_check_cls_and_funtions()

# Run tests when imported by pytest
for name, obj in list(globals().items()):
    if name.startswith('test_') and callable(obj):
        globals()[name] = pytest.mark.doc(obj)
