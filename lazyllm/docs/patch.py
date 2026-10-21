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

``LazyPatchLoader`` 是一个导入系统加载器，它在模块执行时自动为注册在 PATCHS 中的模块应用补丁（目前为 httpx）。
这个加载器包装了原始的模块规范，在模块执行完成后自动调用补丁函数。

Args:
    original_spec (ModuleSpec): 原始模块的规范对象，包含模块的加载信息和路径。
    package_name (str): PATCHS 中的模块名称。

功能:

- 在模块加载时自动设置正确的包和路径属性
- 执行原始加载器的模块执行逻辑
- 在模块执行完成后自动应用对应模块的注册补丁

""")

add_english_doc('LazyPatchLoader', """\
Lazy Patch Loader for automatically applying patches during module loading.

The ``LazyPatchLoader`` is an import system loader that automatically applies patches 
to a module registered in PATCHS (currently httpx) when that module is executed. This loader wraps the
original module specification and automatically calls patch functions after module execution.

Args:
    original_spec (ModuleSpec): The original module's specification object containing 
                                module loading information and paths.
    package_name (str): Module name registered in PATCHS.

Features:

- Automatically sets correct package and path attributes during module loading
- Executes the original loader's module execution logic
- Automatically applies the registered patch for the module after execution

""")

add_chinese_doc('LazyPatchLoader.exec_module', """\
执行模块的加载和初始化过程。

此方法是导入系统加载器的核心方法，负责执行模块的代码并初始化模块对象。
在LazyPatchLoader中，此方法会先设置模块的包和路径属性，然后执行原始加载器的模块执行逻辑，
最后根据 package_name 调用 PATCHS 中对应的补丁函数。

Args:
    module (ModuleType): 要执行的模块对象。

""")

add_english_doc('LazyPatchLoader.exec_module', """\
Execute the module loading and initialization process.

This method is the core method of the import system loader, responsible for executing 
the module's code and initializing the module object. In LazyPatchLoader, this method 
first sets the module's package and path attributes, then executes the original loader's 
module execution logic, and finally calls the patch registered in PATCHS for package_name.

Args:
    module (ModuleType): The module object to be executed.

""")
add_chinese_doc('LazyPatchFinder', """\
延迟补丁查找器，用于在导入时拦截特定模块并应用补丁。

``LazyPatchFinder`` 是一个元路径查找器，它在导入过程中拦截 PATCHS 中尚未打补丁的模块（目前为 httpx），
并使用自定义的LazyPatchLoader来加载这些模块，从而在模块加载时自动应用补丁。

**Note:**\n
- 此查找器处理注册模块尚未打补丁时的导入
- lazyllm.patch 初始化时会直接为已导入的 httpx 调用 patch_httpx()
- requests 在 lazyllm.patch 初始化时直接导入并打补丁，不经过此查找器
- 支持模块的原始属性和路径保持完整
""")

add_english_doc('LazyPatchFinder', """\
Lazy Patch Finder for intercepting specific module imports and applying patches.

The ``LazyPatchFinder`` is a meta path finder that intercepts import requests for 
unpatched modules registered in PATCHS (currently httpx), and uses a custom
LazyPatchLoader to load these modules, automatically applying patches during module loading.

**Note:**\n
- This finder handles imports of registered modules that have not yet been patched
- When lazyllm.patch initializes, an already imported httpx is patched directly with patch_httpx()
- requests is imported and patched directly during lazyllm.patch initialization, outside this finder
- Maintains the original attributes and paths of the modules
""")
add_chinese_doc('LazyPatchFinder.find_spec', """\
查找并返回模块的规范对象，用于自定义模块加载过程。

此方法是MetaPathFinder的核心方法，负责在导入过程中查找指定模块的规范对象。
在LazyPatchFinder中，它只拦截 PATCHS 中尚未打补丁的模块（目前为 httpx），使用自定义的LazyPatchLoader来包装原始模块规范。

Args:
    fullname (str): 要导入的完整模块名称
    path (list): 搜索路径列表，对于顶级模块为None
    target (module, optional): 目标模块对象（重载时使用）

**Returns:**\n
- 对于 PATCHS 中尚未打补丁且能找到原始规范的模块：返回使用 LazyPatchLoader 包装的规范
- 对于其他模块、已打补丁的模块或找不到原始规范的模块：返回 None

""")

add_english_doc('LazyPatchFinder.find_spec', """\
Find and return the module specification object for custom module loading process.

This method is the core method of MetaPathFinder, responsible for finding the specification 
object of the specified module during the import process. In LazyPatchFinder, it specifically 
intercepts unpatched modules registered in PATCHS (currently httpx), using a custom LazyPatchLoader
to wrap the original module specification.

Args:
    fullname (str): The full name of the module to import
    path (list): Search path list, None for top-level modules
    target (module, optional): Target module object (used during reloading)

**Returns:**\n
- For unpatched registered modules with an original specification: Returns the specification wrapped with LazyPatchLoader
- For other modules, already patched modules, or missing original specifications: Returns None

""")
