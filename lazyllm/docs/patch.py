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
add_chinese_doc('LazyPatchFinder', """\
延迟补丁查找器，用于在导入时拦截特定模块并应用补丁。

``LazyPatchFinder`` 是一个元路径查找器，它在导入过程中拦截对'requests'和'httpx'模块的导入请求，
并使用自定义的LazyPatchLoader来加载这些模块，从而在模块加载时自动应用补丁。

**Note:**\n
- 此查找器只在模块尚未导入时生效
- 如果模块已经导入，会直接调用patch_requests_and_httpx()函数
- 支持模块的原始属性和路径保持完整
""")

add_english_doc('LazyPatchFinder', """\
Lazy Patch Finder for intercepting specific module imports and applying patches.

The ``LazyPatchFinder`` is a meta path finder that intercepts import requests for 
'requests' and 'httpx' modules during the import process, and uses a custom 
LazyPatchLoader to load these modules, automatically applying patches during module loading.

**Note:**\n
- This finder only takes effect when modules are not already imported
- If modules are already imported, directly calls patch_requests_and_httpx() function
- Maintains the original attributes and paths of the modules
""")
add_chinese_doc('LazyPatchFinder.find_spec', """\
查找并返回模块的规范对象，用于自定义模块加载过程。

此方法是MetaPathFinder的核心方法，负责在导入过程中查找指定模块的规范对象。
在LazyPatchFinder中，它专门拦截'requests'和'httpx'模块的导入请求，使用自定义的LazyPatchLoader来包装原始模块规范。

Args:
    fullname (str): 要导入的完整模块名称
    path (list): 搜索路径列表，对于顶级模块为None
    target (module, optional): 目标模块对象（重载时使用）

**Returns:**\n
- 对于'requests'和'httpx'模块：返回使用LazyPatchLoader包装的模块规范
- 对于其他模块：返回None，让其他查找器继续处理

""")

add_english_doc('LazyPatchFinder.find_spec', """\
Find and return the module specification object for custom module loading process.

This method is the core method of MetaPathFinder, responsible for finding the specification 
object of the specified module during the import process. In LazyPatchFinder, it specifically 
intercepts import requests for 'requests' and 'httpx' modules, using a custom LazyPatchLoader 
to wrap the original module specification.

Args:
    fullname (str): The full name of the module to import
    path (list): Search path list, None for top-level modules
    target (module, optional): Target module object (used during reloading)

**Returns:**\n
- For 'requests' and 'httpx' modules: Returns module specification wrapped with LazyPatchLoader
- For other modules: Returns None, allowing other finders to continue processing

""")
