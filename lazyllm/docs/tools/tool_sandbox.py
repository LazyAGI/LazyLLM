# flake8: noqa E501
import importlib
import functools
from .. import utils
add_sandbox_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.sandbox'))
add_sandbox_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.sandbox'))
add_sandbox_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.sandbox'))

add_sandbox_chinese_doc('LazyLLMSandboxBase', '''\
沙箱执行基类，定义统一的代码执行接口与语言检查逻辑。

Args:
    output_dir_path (str | None): 输出文件保存目录，默认当前工作目录，可能会覆盖当前工作目录下的文件。
    return_trace (bool): 是否返回中间执行信息（由 ModuleBase 控制）。

Notes:
    子类需实现 `_is_available` 与 `_execute` 方法。
''')

add_sandbox_english_doc('LazyLLMSandboxBase', '''\
Base class for sandbox execution with a unified call interface and language validation.

Args:
    output_dir_path (str | None): output directory for generated files, default is cwd.
    return_trace (bool): whether to return intermediate execution info (controlled by ModuleBase).

Notes:
    Subclasses must implement `_is_available` and `_execute`.
''')

add_sandbox_chinese_doc('LazyLLMSandboxBase.forward', '''\
统一执行入口，负责语言校验并调用具体实现。

Args:
    code (str): 待执行的代码。
    language (str): 代码语言，默认 'python'。
    input_files (list[str] | None): 输入文件路径列表，可选。
    output_files (list[str] | None): 需要回传的输出文件列表，可选。

**Returns:**\n
    由具体沙箱实现返回的结果（通常为 dict 或错误信息字符串）。
''')

add_sandbox_english_doc('LazyLLMSandboxBase.forward', '''\
Unified execution entry that validates language and delegates to the implementation.

Args:
    code (str): code to execute.
    language (str): code language, default 'python'.
    input_files (list[str] | None): optional list of input file paths.
    output_files (list[str] | None): optional list of output files to fetch.

**Returns:**\n
    Result produced by the sandbox implementation (usually a dict or an error message string).
''')

add_sandbox_chinese_doc('DummySandbox', '''\
本地沙箱实现（python-only），用于在受限环境中执行代码。

特点：
- 通过 AST + SecurityVisitor 做基础安全检查。
- 在临时目录中运行代码，执行完毕后清理。
- 返回 stdout/stderr/returncode 的字典结果。

Args:
    timeout (int): 超时时间（秒），默认 30。
    project_dir (str | None): 若指定，将项目内 .py 文件复制到沙箱执行目录，便于引用。
    return_trace (bool): 是否返回中间执行信息。
''')

add_sandbox_english_doc('DummySandbox', '''\
Local sandbox implementation (python-only) for executing code in a restricted environment.

Features:
- Basic safety checks with AST + SecurityVisitor.
- Runs code in a temp directory and cleans up afterwards.
- Returns a dict with stdout/stderr/returncode.

Args:
    timeout (int): timeout in seconds, default 30.
    project_dir (str | None): if provided, copies .py files into sandbox for imports.
    return_trace (bool): whether to return intermediate execution info.
''')

add_sandbox_example('DummySandbox', """\
>>> from lazyllm.tools.sandbox import DummySandbox
>>> sandbox = DummySandbox(timeout=10)
>>> result = sandbox(code="print(1 + 1)")
>>> print(result['stdout'].strip())
2
""")

add_sandbox_chinese_doc('SandboxFusion', '''\
远程沙箱实现，通过 HTTP API 执行代码并获取结果。

支持语言：python / bash。可配置编译超时、运行超时、内存限制，并支持上传工程文件与拉取输出文件。

Args:
    base_url (str): 远程沙箱服务地址，默认来自 config['sandbox_fusion_base_url']。
    compile_timeout (int): 编译超时（秒），默认 10。
    run_timeout (int): 运行超时（秒），默认 10。
    memory_limit_mb (int): 内存限制（MB），-1 表示不限制。
    project_dir (str | None): 若指定，将工程目录下的 .py 文件上传到沙箱。

Notes:
    需要配置 LAZYLLM_SANDBOX_FUSION_BASE_URL 或显式传入 base_url。
''')

add_sandbox_english_doc('SandboxFusion', '''\
Remote sandbox implementation that executes code via HTTP API.

Supports python / bash. Configurable compile/run timeouts and memory limits. Can upload project files and fetch output files.

Args:
    base_url (str): remote sandbox base URL, defaults to config['sandbox_fusion_base_url'].
    compile_timeout (int): compile timeout in seconds, default 10.
    run_timeout (int): run timeout in seconds, default 10.
    memory_limit_mb (int): memory limit in MB, -1 means no limit.
    project_dir (str | None): if provided, uploads .py files from the project directory.

Notes:
    Set LAZYLLM_SANDBOX_FUSION_BASE_URL or pass base_url explicitly.
''')

add_sandbox_example('SandboxFusion', """\
>>> from lazyllm import config
>>> from lazyllm.tools.sandbox import SandboxFusion
>>> config['sandbox_fusion_base_url'] = "http://localhost:8000"
>>> sandbox = SandboxFusion(run_timeout=5)
>>> result = sandbox(code="print('ok')")
>>> print(result['stdout'].strip())
ok
""")

