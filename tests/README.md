# 中文

## 测试分类

测试分为 `basic_tests` , `charge_tests` , `advanced_tests/standard` , `advanced_tests/full` 四类。

- `basic_tests` : 基本的测试，主要测一些基本功能
- `charge_tests` : 需要使用线上模型的测例，需要付费
- `advanced_tests/standard` : 需要本地部署大模型的测例，使用默认的推理框架
- `advanced_tests/full` : 需要本地部署大模型的测例，使用非默认的推理框架

此外，测试会分别在 `linux` , `windows` , `macos` 上执行。

- `linux` : 执行全量的测试
- `windows` : 仅执行 `basic_tests` 和 `charge_tests` ，并且过滤其中的 `test_engine`
- `macos` : 仅执行 `basic_tests` 和 `charge_tests` ，并且过滤其中的 `test_engine`

## 缓存策略

- main分支对本地模型、在线模型进行缓存的读写，包括大模型、向量模型、重排序模型、多模态模型等。具体策略为:
  1. 在执行CI任务时下载旧的缓存文件，基于旧的缓存文件执行测例。在linux为读写模式，其他为只读模式。
  2. 在linux上，测例全部执行成功后，上传缓存文件到github，覆盖原缓存文件。
  3. 第一次执行需要部署的测例时，禁止模型部署，尝试从缓存中读取结果，如果没有读取成功，则报错，进入rerun
  4. 在rerun的时候，检测异常，如果是CacheNotFound，则部署模型，并且临时禁用读缓存操作，计算结果写入缓存。
  5. 当发现测例全部通过之后，提交缓存文件到云端。
- 非main分支的pr在执行CI任务时下载缓存，以只读的形式使用。具体策略为:
  1. 在执行CI任务时下载旧的缓存文件，基于旧的缓存文件执行测例，模式为只读。
  2. 第一次执行需要部署的测例时，禁止模型部署，尝试从缓存中读取结果，如果没有读取成功，则报错，进入rerun
  3. 在rerun的时候，检测异常，如果是CacheNotFound，则部署模型，并且禁用读缓存操作。
  

# English
## Test Categories

Tests are divided into four categories: `basic_tests`, `charge_tests`, `advanced_tests/standard`, and `advanced_tests/full`.

- `basic_tests`: Basic tests that primarily test fundamental functionality
- `charge_tests`: Test cases that require online models and incur costs
- `advanced_tests/standard`: Test cases that require local deployment of large models using the default inference framework
- `advanced_tests/full`: Test cases that require local deployment of large models using non-default inference frameworks

Additionally, tests are executed on `linux`, `windows`, and `macos` respectively.

- `linux`: Executes all tests
- `windows`: Only executes `basic_tests` and `charge_tests`, filtering out `test_engine` from them
- `macos`: Only executes `basic_tests` and `charge_tests`, filtering out `test_engine` from them

## Caching Strategy

- The main branch performs read/write caching for local models and online models, including large models, vector models, reranking models, multimodal models, etc. The specific strategy is:
  1. Download old cache files when executing CI tasks and run test cases based on the old cache files. Linux uses read/write mode, while others use read-only mode.
  2. On Linux, after all test cases pass successfully, upload cache files to GitHub, overwriting the original cache files.
  3. When executing test cases that require deployment for the first time, disable model deployment, attempt to read results from cache, and if unsuccessful, report an error and enter rerun.
  4. During rerun, detect exceptions, and if it's CacheNotFound, deploy the model and temporarily disable read cache operations, writing calculation results to cache.
  5. After all test cases pass, commit cache files to the cloud.
- Non-main branch PRs download cache when executing CI tasks and use it in read-only mode. The specific strategy is:
  1. Download old cache files when executing CI tasks and run test cases based on the old cache files in read-only mode.
  2. When executing test cases that require deployment for the first time, disable model deployment, attempt to read results from cache, and if unsuccessful, report an error and enter rerun.
  3. During rerun, detect exceptions, and if it's CacheNotFound, deploy the model and disable read cache operations.
