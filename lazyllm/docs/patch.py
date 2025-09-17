# flake8: noqa E501
from . import utils
import functools
import lazyllm

# ============= Patch

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.patch)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.patch)
add_example = functools.partial(utils.add_example, module=lazyllm.patch)

# LazyLLMPatch
add_chinese_doc('LazyPatchLoader', """\
延迟补丁加载器，用于在模块加载时自动应用补丁。

``LazyPatchLoader`` 是一个导入系统加载器，它在模块执行时自动为请求库和httpx库应用补丁。
这个加载器包装了原始的模块规范，在模块执行完成后自动调用补丁函数。

Args:
    original_spec (ModuleSpec): 原始模块的规范对象，包含模块的加载信息和路径。

功能:

- 在模块加载时自动设置正确的包和路径属性
- 执行原始加载器的模块执行逻辑
- 在模块执行完成后自动应用requests和httpx库的补丁

""")

add_english_doc('LazyPatchLoader', """\
Lazy Patch Loader for automatically applying patches during module loading.

The ``LazyPatchLoader`` is an import system loader that automatically applies patches 
to the requests and httpx libraries when a module is executed. This loader wraps the 
original module specification and automatically calls patch functions after module execution.

Args:
    original_spec (ModuleSpec): The original module's specification object containing 
                                module loading information and paths.

Features:

- Automatically sets correct package and path attributes during module loading
- Executes the original loader's module execution logic
- Automatically applies patches to requests and httpx libraries after module execution

""")

add_chinese_doc('LazyPatchLoader.exec_module', """\
执行模块的加载和初始化过程。

此方法是导入系统加载器的核心方法，负责执行模块的代码并初始化模块对象。
在LazyPatchLoader中，此方法会先设置模块的包和路径属性，然后执行原始加载器的模块执行逻辑，
最后自动为requests和httpx库应用补丁。

Args:
    module (ModuleType): 要执行的模块对象。

""")

add_english_doc('LazyPatchLoader.exec_module', """\
Execute the module loading and initialization process.

This method is the core method of the import system loader, responsible for executing 
the module's code and initializing the module object. In LazyPatchLoader, this method 
first sets the module's package and path attributes, then executes the original loader's 
module execution logic, and finally automatically applies patches to the requests and 
httpx libraries.

Args:
    module (ModuleType): The module object to be executed.

""")
