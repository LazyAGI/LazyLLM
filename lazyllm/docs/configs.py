# flake8: noqa E501
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm)
add_example = functools.partial(utils.add_example, module=lazyllm)


add_chinese_doc('Config', '''\
Config是LazyLLM提供的配置类，可以支持通过加载配置文件、设置环境变量、编码写入默认值等方式设置LazyLLM框架的相关配置，以及导出当前所有的配置项和对应的值。
Config模块自动生成一个config对象，其中包含所有的配置。

Args:
    prefix (str, optional): 环境变量前缀。默认为 'LAZYLLM'
    home (str, optional): 配置文件目录路径。默认为 '~/.lazyllm'
''')


add_english_doc('Config', '''\
Config is a configuration class provided by LazyLLM, which loads configurations of LazyLLM framework from config files,
environment variables, or specify them explicitly. it can export all configuration items as well.
The Config module automatically generates an object named 'config' containing all configurations.

Args:
    prefix (str, optional): Environment variable prefix. Defaults to 'LAZYLLM'
    home (str, optional): Configuration file directory path. Defaults to '~/.lazyllm'
''')


add_chinese_doc('Config.done', '''\
用于检查config.json配置文件中是否还有没有通过add方法载入的配置项
''')

add_chinese_doc('Config.getenv', '''\
用于检查config.json配置文件中是否还有没有通过add方法载入的配置项
Config.getenv(name, type, default): -> str\n
获取LazyLLM相关环境变量的值

Args:
    name (str): 不包含前缀的环境变量名字，不区分大小写。函数将从拼接了前缀和此名字的全大写的环境变量中获取对应的值。
    type (type): 指定该配置的类型，例如str。对于bool型，函数会将'TRUE', 'True', 1, 'ON', '1'这5种输入转换为True。
    default (可选): 若无法获取到环境变量的值，将返回此变量。
''')

add_chinese_doc('Config.add', '''\
将值加载至LazyLLM的配置项中。函数首先尝试从加载自config.json的字典中查找名字为name的值。若找到则从该字典中删去名为name的键。并将对应的值写入config。
若env是一个字符串，函数会调用getenv寻找env对应的LazyLLM环境变量，若找到则写入config。如果env为一个字典，函数将尝试调用getenv寻找字典中的key对于的环境变量并转换为bool型。
若转换完成的bool值是True，则将字典中当前的key对应的值写入config。

Args:
    name (str): 配置项名称
    type (type): 该配置的类型
    default (可选): 若无法获取到任何值，该配置的默认值
    env (可选): 不包含前缀的环境变量名称，或者一个字典，其中的key是不包含前缀的环境变量名称，value是要加入配置的值。
''')

add_chinese_doc('Config.get_all_configs', '''\
获取config中所有的配置
''')

add_english_doc('Config.done', '''\
Check if any configuration items in the config.json file that is not loaded by the add method.

Args:
    None.
''')

add_english_doc('Config.getenv', '''\
Get value of LazyLLM-related environment variables.

Args:
    name (str): The name of the environment variable （without the prefix）, case-insensitive. The function obtains value
    from environment variable by concatenating the prefix and this name, with all uppercase letters.
    type (type): Specifies the type of the configuration, for example, str. For boolean types, the function will
    convert inputs ‘TRUE’, ‘True’, 1, ‘ON’, and ‘1’ to True.
    default (optional): If the value of the environment variable cannot be obtained, this value is returned.
''')

add_english_doc('Config.add', '''\
Loads value into LazyLLM configuration item. The function first attempts to find the value with the given name from the
dict loaded from config.json. If found, it removes the key from the dict and saves the value to the config.
If 'env' is a string, the function calls getenv to look for the corresponding LazyLLM environment variable, and if
it's found, writes it to the config. If 'env' is a dictionary, the function attempts to call getenv to find the
environment variables corresponding to the keys in the dict and convert them to boolean type.
If the converted boolean value is True, the value corresponding to the current key in the dict is written to the config.

Args:
    name (str): The name of the configuration item
    type (type): The type of the configuration
    default (optional): The default value of the configuration if no value can be obtained
    env (optional): The name of the environment variable without the prefix, or a dictionary where the keys are the
    names of the environment variables(without the prefix), and the values are what to be added to the configuration.
''')

add_english_doc('Config.get_all_configs', '''\
Get all configurations from the config.

Args:
    None.
''')


add_example('Config.get_all_configs', '''\
>>> import lazyllm
>>> from lazyllm.configs import config
>>> config['launcher']
'empty'
>>> config.get_all_configs()
{'home': '~/.lazyllm/', 'mode': <Mode.Normal: (1,)>, 'repr_ml': False, 'rag_store': 'None', 'redis_url': 'None', ...}
''')

add_chinese_doc('Config.get_config', '''\ 
静态方法：从配置字典中获取配置。
这是一个简单的配置获取方法，主要用于从已加载的配置字典中提取配置信息。

Args:
    cfg (dict): 从配置文件中读取的配置字典。
''')

add_english_doc('Config.get_config', '''
Static method: Get configuration from config dictionary.
This is a simple configuration retrieval method mainly used to extract configuration information from already loaded configuration dictionaries.

Args:
    cfg (dict): The configuration dictionary read from the config file.
''')

add_chinese_doc('Config.temp', '''
临时修改配置项的上下文管理器。在with语句块内临时修改指定配置项的值，退出语句块后自动恢复原值。注意，此函数并非线程安全，请勿在多线程或者多协程的环境下使用。

Args:
    name (str): 要临时修改的配置项名称。
    value (Any): 临时设置的值。
''')

add_english_doc('Config.temp', '''
Context manager for temporary configuration modification.
Temporarily modifies the value of the specified configuration item within the with statement block, and automatically restores the original value when exiting the block.
Attention: this function is not thread-safe, you should not use it in multi-thread or multi-coroutine environment.

Args:
    name (str): The name of the configuration item to temporarily change.
    value (Any): The temporary value to set.
''')

add_chinese_doc('Config.refresh', '''
根据环境变量的最新值刷新配置项。如果传入 targets 为字符串，则按单个配置项更新；如果为列表，则批量更新；如果为 None，则扫描所有已映射到环境变量的配置项并更新。

Args:
    targets (str | list[str] | None): 要刷新的配置项名称或列表，传 None 表示刷新所有可从环境变量读取的项。
''')

add_english_doc('Config.refresh', '''
Refresh configuration items based on the latest environment variable values.  
If `targets` is a string, updates the single corresponding configuration item;  
if it's a list, updates multiple;  
if None, scans all environment-variable-mapped configuration items and updates them.

Args:
    targets (str | list[str] | None): Name of the config key or list of keys to refresh, or None to refresh all environment-backed keys.
''')

add_chinese_doc('namespace', '''\
命名空间包装器，用于在指定的配置命名空间（namespace）中调用 LazyLLM 的模块构造函数。

`namespace` 既可以作为上下文管理器使用，也可以直接通过属性访问的方式，
在不显式使用 `with` 的情况下，将某一次模块构造绑定到指定的 namespace 中。

支持的模块包括：
AutoModel、OnlineModule、OnlineChatModule、OnlineEmbeddingModule、OnlineMultiModalModule。

**用法说明：**\n
- 作为上下文管理器：在 `with lazyllm.namespace(space)` 块内，所有 LazyLLM 配置和模块构造
  都会使用对应的 namespace。
- 作为包装器调用：通过 `lazyllm.namespace(space).OnlineChatModule(...)` 的形式，
  仅对单次模块构造生效，不影响全局状态。

**注意事项：**\n
- `namespace` 实例本身不是线程安全的，多线程环境中应为每个线程创建独立实例。
''')

add_english_doc('namespace', '''\
A namespace wrapper used to invoke LazyLLM module constructors under a specified configuration namespace.

`namespace` can be used either as a context manager or as a lightweight wrapper for single calls.
It allows binding LazyLLM configuration and module construction to a specific namespace
without affecting the global configuration.

Supported modules include:
AutoModel, OnlineModule, OnlineChatModule, OnlineEmbeddingModule, and OnlineMultiModalModule.

**Usage:**\n
- As a context manager: within a `with lazyllm.namespace(space)` block, all LazyLLM configuration
  and module construction will use the given namespace.
- As a wrapper call: using `lazyllm.namespace(space).OnlineChatModule(...)` applies the namespace
  only to that single constructor call.

**Notes:**\n
- A `namespace` instance is not thread-safe. In multi-threaded environments,
  create a separate instance per thread even if they share the same space name.
''')

add_example('namespace', '''\
>>> import os
>>> import lazyllm
>>> from lazyllm import namespace
>>> with lazyllm.namespace('my'):
...     assert lazyllm.config['gpu_type'] == 'A100'
...     os.environ['MY_GPU_TYPE'] = 'H100'
...     assert lazyllm.config['gpu_type'] == 'H100'
...
>>>
>>> assert lazyllm.config['gpu_type'] == 'A100'
>>>
>>> with lazyllm.namespace('my'):
...     m = lazyllm.OnlineChatModule()
...
>>> m = lazyllm.namespace('my').OnlineChatModule()
''')

add_chinese_doc('namespace.register_module', """\
向 `namespace` 注册可被代理调用的 LazyLLM 模块名称。

被注册的模块名将被加入 `namespace.supported` 集合中，
之后即可通过 `namespace(space).<ModuleName>(...)` 的形式，
在指定的 namespace 下构造对应模块。

该方法是一个类级别的注册接口，对所有 `namespace` 实例生效。

**Parameters:**\n
- module (str | List[str]): 需要注册的模块名称。
  - 当为字符串时，注册单个模块名；
  - 当为列表时，批量注册多个模块名。
""")

add_english_doc('namespace.register_module', """\
Register LazyLLM module names that can be proxied by `namespace`.

Registered module names will be added to the class-level `namespace.supported` set,
allowing them to be constructed via `namespace(space).<ModuleName>(...)`
under the specified namespace.

This is a class-level registration method and affects all `namespace` instances.

**Parameters:**\n
- module (str | List[str]): The module name(s) to register.
  - A string registers a single module name;
  - A list of strings registers multiple module names at once.
""")

add_example('namespace.register_module', """\
>>> import lazyllm
>>> from lazyllm import namespace
>>> namespace.register_module('OnlineChatModule')
>>> 'OnlineChatModule' in namespace.supported
True
>>> namespace.register_module(['AutoModel', 'OnlineEmbeddingModule'])
>>> 'AutoModel' in namespace.supported
True
>>> 'OnlineEmbeddingModule' in namespace.supported
True
>>> namespace('my').OnlineChatModule().space
'my'
""")
