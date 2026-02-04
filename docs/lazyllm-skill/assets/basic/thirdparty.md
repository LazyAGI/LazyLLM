# Thirdparty使用示例

## 替代import的使用

在 LazyLLM 代码里，**不要**直接 `import torch`、`import pandas`，而应从 `lazyllm.thirdparty` 导入，这样才走懒加载和统一报错：

```python
from lazyllm.thirdparty import torch, transformers
from lazyllm.thirdparty import pandas as pd
from lazyllm.thirdparty import numpy as np
from lazyllm.thirdparty import PIL
from lazyllm.thirdparty import gradio as gr
```

- 仅做 `from lazyllm.thirdparty import torch` 时不会执行 `import torch`。
- 第一次使用 `torch.nn`、`torch.tensor(...)` 等时，才会 `import torch`；若未安装，会得到带 `pip install torch`（或映射后的包名+版本）的 `ImportError`。

子模块写法与直接 import 一致，例如：

```python
from lazyllm.thirdparty import PIL
img = PIL.Image.open('x.png')

from lazyllm.thirdparty import fsspec
fs = fsspec.implementations.local.LocalFileSystem()
```

只要在 `modules` 里配置了 `['PIL', 'Image']`、`['fsspec', 'implementations.local']` 等，就会按路径懒加载到对应子模块。

## 在入口做"按组"依赖校验

某些模块（如 RAG、SQL）要求一整组依赖安装完整，可以在包或模块的 `__init__.py` 里用 `check_dependency_by_group` 在导入时强制校验：

```python
# lazyllm/tools/rag/__init__.py
from lazyllm.thirdparty import check_dependency_by_group
check_dependency_by_group('rag')
```

- `group_name` 对应 `pyproject.toml` 里 `[tool.poetry.extras]` 下的 `rag`、`standard`、`full` 等。
- 若该组中有包未安装，会 `raise ImportError`，并提示：`lazyllm install rag`（或对应 extra 名）。

这样用户只要 `import lazyllm.tools.rag`，就能在缺依赖时立刻拿到明确指引。

## 在运行时做“按包”检查和提示

不需要按 extra 组，只关心若干具体包时，用 `check_packages`：

```python
import lazyllm.thirdparty as thirdparty

def need_pptx():
    thirdparty.check_packages(['python-pptx', 'torch', 'Pillow', 'transformers'])
    # 若缺任一，会 LOG.warning 并打印 pip install 命令，不抛异常
```

- `check_packages` 不会 `raise`，只打日志和安装命令，适合“可选增强”场景。
- 若要用 **import 名**（如 `sklearn`），`get_pip_install_cmd` 会通过 `package_name_map` 转成 `scikit-learn` 等 pip 名。

