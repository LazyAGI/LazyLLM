# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.patch)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.patch)
add_example = functools.partial(utils.add_example, module=lazyllm.patch)

add_chinese_doc('LazyPatchFinder.find_spec', """\
实现 importlib.abc.MetaPathFinder 接口的模块查找方法。

当尝试导入模块时被调用。若模块名为 'requests' 或 'httpx'，则临时移除当前查找器，
使用原始加载器获取模块 spec，并将其替换为由 LazyPatchLoader 包装的新 spec，
从而实现对目标模块的透明补丁注入。处理完成后，该查找器不再参与后续导入。

Args:
    fullname (str): 要导入的完整模块名
    path (Optional[Sequence[str]]): 模块搜索路径（对顶层模块为 None）
    target (Optional[types.ModuleType]): 用于相对导入的目标模块（通常可忽略）

Returns:
    ModuleSpec 或 None：若需拦截该模块则返回新构造的 spec，否则返回 None
""")

add_english_doc('LazyPatchFinder.find_spec', """\
Implements the find_spec method of importlib.abc.MetaPathFinder.

Called when a module is being imported. If the module name is 'requests' or 'httpx',
it temporarily removes itself from sys.meta_path, retrieves the original module spec,
and replaces it with a new spec wrapped by LazyPatchLoader to enable transparent patching.
After handling, this finder will not interfere with subsequent imports.

Args:
    fullname (str): The fully qualified name of the module being imported
    path (Optional[Sequence[str]]): The module search path (None for top-level modules)
    target (Optional[types.ModuleType]): Target module for relative imports (usually ignored)

Returns:
    ModuleSpec or None: Returns a patched ModuleSpec if the module should be intercepted, otherwise None
""")