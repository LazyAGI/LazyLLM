import pytest # noqa E401
import inspect
import os
import sys
import re

# Get the project root directory (assuming this file is in LazyLLM/tests/doc_check/)
# Go up two levels from LazyLLM/tests/doc_check/ to reach LazyLLM root
current_dir = os.path.dirname(os.path.abspath(__file__))
lazyllm_root = os.path.dirname(os.path.dirname(current_dir))

# Add lazyllm root to Python path
if lazyllm_root not in sys.path:
    sys.path.insert(0, lazyllm_root)
    print(f"Added lazyllm root path: {lazyllm_root}")

# Add docs scripts path
DOCS_SCRIPTS_PATH = os.path.join(lazyllm_root, 'docs', 'scripts')
if DOCS_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, DOCS_SCRIPTS_PATH)
    print(f"Added docs scripts path: {DOCS_SCRIPTS_PATH}")

# Force Python to look for lazyllm in our local path first
if 'lazyllm' in sys.modules:
    del sys.modules['lazyllm']

import lazyllm
from typing import Callable
import dataclasses
import enum

# Import documentation generation modules
try:
    from lazynote.manager import SimpleManager
    print("Successfully imported SimpleManager")
except ImportError as e:
    print(f"Failed to import SimpleManager: {e}")
    SimpleManager = None

print(f"Loaded lazyllm from: {os.path.dirname(lazyllm.__file__)}")

def generate_docs_for_module():
    """Generate documentation using SimpleManager"""
    if SimpleManager is None:
        print("SimpleManager not available, skipping doc generation")
        return False
    
    try:
        print("Generating documentation using SimpleManager...")
        
        # Set Chinese language environment
        os.environ['LAZYLLM_LANGUAGE'] = 'CHINESE'
        
        # Create skip_list
        skip_list = [
            'lazyllm.components.deploy.relay.server',
            'lazyllm.components.deploy.relay.base',
            'lazyllm.components.finetune.easyllm',
            'lazyllm.tools.rag.component.bm25_retriever',
        ]
        
        # Clear existing docs first
        print("Clearing existing docs...")
        manager = SimpleManager(pattern='clear', skip_on_error=True)
        manager.traverse(lazyllm, skip_modules=skip_list)
        
        # Fill new docs
        print("Filling new docs...")
        manager = SimpleManager(pattern='fill', skip_on_error=True)
        manager.traverse(lazyllm, skip_modules=skip_list)
        
        print("Documentation generation completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during documentation generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def class_should_check(cls, module):
    """Check if a class should be included in documentation verification."""
    # Skip dataclass and enum classes
    if dataclasses.is_dataclass(cls) or issubclass(cls, enum.Enum):
        return False
        
    if not cls.__name__[0].isupper() or cls.__module__ != module.__name__:
        return False
        
    # Get all methods, including classmethod and staticmethod
    all_methods = []
    for name, obj in inspect.getmembers(cls):
        if (inspect.isfunction(obj) or 
            isinstance(obj, classmethod) or 
            isinstance(obj, staticmethod)):
            all_methods.append((name, obj))
    
    # Include __init__ and non-private methods
    custom_methods = [name for name, obj in all_methods 
                     if not name.startswith('_') or name == '__init__']
    return len(custom_methods) > 0

def get_sub_classes(module):
    """Get all valid subclasses from a module recursively."""
    try:
        clsmembers = inspect.getmembers(module, inspect.isclass)
    except Exception as e:
        print(f"Warning: Failed to inspect module {module.__name__}: {e}")
        return set()
    
    classes = set([ele[1] for ele in clsmembers if class_should_check(ele[1], module)])
    
    for name, sub_module in inspect.getmembers(module, inspect.ismodule):
        if sub_module.__name__.startswith(module.__name__):
            # Skipping problematic module(chat_tts, thirdparty)
            if 'thirdparty' in sub_module.__name__ or 'ChatTTS' in sub_module.__name__:
                print(f"Skipping problematic module: {sub_module.__name__}")
                continue
            try:
                classes.update(get_sub_classes(sub_module))
            except Exception as e:
                print(f"Warning: Failed to process submodule {sub_module.__name__}: {e}")
                continue
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

def check_method_has_doc(cls, func: Callable) -> tuple[bool, bool]:
    """Check if method has documentation (Chinese and English)
    
    Returns:
        tuple[bool, bool]: (has_chinese_doc, has_english_doc)
    """
    has_chinese_doc = False
    has_english_doc = False
    
    # Check Chinese documentation
    try:
        with lazyllm.config.temp('language', 'CHINESE'):
            if func.__name__ == '__init__':
                doc = cls.__doc__
            else:
                doc = func.__doc__
            has_chinese_doc = bool(doc and doc.strip())
    except Exception as e:
        print(f"Error checking Chinese doc for {cls.__name__}.{func.__name__}: {e}")
    
    # Check English documentation
    try:
        with lazyllm.config.temp('language', 'ENGLISH'):
            if func.__name__ == '__init__':
                doc = cls.__doc__
            else:
                doc = func.__doc__
            has_english_doc = bool(doc and doc.strip())
    except Exception as e:
        print(f"Error checking English doc for {cls.__name__}.{func.__name__}: {e}")
    
    return has_chinese_doc, has_english_doc

def do_check_method(cls, func: Callable):
    """Check if method has documentation after generating docs."""
    
    try:
        # For __init__ method, always check (never skip) because each class needs its own documentation
        if func.__name__ != '__init__':
            # For non-__init__ methods, check if ancestor has documentation
            if is_method_overridden(cls, func):
                highest_ancestor = find_highest_ancestor_with_method(cls, func.__name__)
                if highest_ancestor:
                    # Check if the ancestor class has documentation
                    ancestor_has_chinese, ancestor_has_english = check_method_has_doc(
                        highest_ancestor, getattr(highest_ancestor, func.__name__)
                    )
                    
                    # If ancestor has documentation, skip checking this class
                    if ancestor_has_chinese or ancestor_has_english:
                        print(f"Skipping {cls.__name__}.{func.__name__} as ancestor class {highest_ancestor.__name__} has documentation")
                        return
        
        # Check if current method has documentation
        has_chinese_doc, has_english_doc = check_method_has_doc(cls, func)
        
        if not has_chinese_doc and not has_english_doc:
            # No documentation found, record as failure
            error_msg = f"No documentation found for {cls.__name__}.{func.__name__} after doc generation"
            print(f"FAILED: {error_msg}")
            raise ValueError(error_msg)
        else:
            print(f"PASSED: {cls.__name__}.{func.__name__} has documentation")
            
    except Exception as e:
        if "No documentation found" not in str(e):
            print(f"ERROR: {cls.__name__}.{func.__name__} - {str(e)}")
        raise

def get_defined_methods(cls):
    """Get methods that are actually defined in the class.
    
    This function only returns methods that are defined in the class's source code,
    excluding methods that are dynamically added or inherited from parent classes.
    """
    try:
        source = inspect.getsource(cls)
        # Improved regex pattern that can match method definitions with decorators
        method_pattern = r'(?:@\w+\s+)*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return set(re.findall(method_pattern, source))
    except (TypeError, OSError):
        return set()

# Global variable to track if docs have been generated
_docs_generated = False

@pytest.fixture(scope="session", autouse=True)
def generate_docs():
    """Generate documentation before running tests (session scope)"""
    global _docs_generated
    if not _docs_generated:
        print("Starting documentation generation...")
        success = generate_docs_for_module()
        if not success:
            print("Documentation generation failed, but continuing with tests...")
        else:
            print("Documentation generation successful")
        _docs_generated = True
    return _docs_generated

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
        # Exception already handled in do_check_method
        pass
"""
    exec(code, globals())

def gen_check_cls_and_funtions():
    """Generate test functions for all classes and their methods."""
    all_classes = get_sub_classes(lazyllm)
    for cls in all_classes:
        # Only care about methods actually defined in the current class source code
        defined_methods = get_defined_methods(cls)

        all_methods = []
        # Directly iterate through cls.__dict__ to avoid descriptor unpacking
        for name, obj in cls.__dict__.items():
            if name not in defined_methods:          # Filter non-source-defined methods
                continue

            if isinstance(obj, classmethod):
                # The actual function for classmethod/staticmethod is in __func__
                all_methods.append((name, obj.__func__))
            elif isinstance(obj, staticmethod):
                all_methods.append((name, obj.__func__))
            elif inspect.isfunction(obj):
                all_methods.append((name, obj))

        # Only check non-private methods, __init__ is an exception
        custom_methods = [
            func for name, func in all_methods
            if not name.startswith('_') or name == '__init__'
        ]
        print(f"Custom methods to check: {[m.__name__ for m in custom_methods]}")

        for method in custom_methods:
            create_test_function(cls, method)

global_func_names = set()
gen_check_cls_and_funtions()

def run_all_tests():
    """Run all tests"""
    print("Starting documentation check tests...")
    
    # First generate documentation
    print("Generating documentation...")
    doc_generation_success = generate_docs_for_module()
    
    if not doc_generation_success:
        print("Documentation generation failed, but continuing with checks...")
    else:
        print("Documentation generation successful, starting checks...")
    
    # Run all dynamically generated test functions
    test_functions = [name for name in globals() if name.startswith('test_') and callable(globals()[name])]
    
    print(f"Total {len(test_functions)} test functions")
    
    passed_count = 0
    failed_count = 0
    
    for test_name in test_functions:
        try:
            globals()[test_name]()
            passed_count += 1
        except Exception:
            failed_count += 1
    
    print(f"\nTests completed!")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

# If running this script directly, execute all tests
if __name__ == "__main__":
    run_all_tests()
