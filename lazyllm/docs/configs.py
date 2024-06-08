from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm)
add_example = functools.partial(utils.add_example, module=lazyllm)


add_chinese_doc('Config', r'''\
Config是LazyLLM提供的配置类，可以支持通过加载配置文件、设置环境变量、编码写入默认值等方式设置LazyLLM框架的相关配置，以及导出当前所有的配置项和对应的值。
Config模块自动生成一个config对象，其中包含所有的配置。

.. function:: Config.done() -> None
用于检查config.json配置文件中是否还有没有通过add方法载入的配置项

参数：
    无

.. function:: Config.getenv(name, type, default): -> str
获取LazyLLM相关环境变量的值

参数:
    name (str): 不包含前缀的环境变量名字，不区分大小写。函数将从拼接了前缀和此名字的全大写的环境变量中获取对应的值。
    type (type): 指定该配置的类型，例如str。对于bool型，函数会将'TRUE', 'True', 1, 'ON', '1'这5种输入转换为True。
    default (可选): 若无法获取到环境变量的值，将返回此变量。

.. function:: Config.add(name, type, default, env):: -> Config
将值加载至LazyLLM的配置项中。函数首先尝试从加载自config.json的字典中查找名字为name的值。若找到则从该字典中删去名为name的键。并将对应的值写入config。
若env是一个字符串，函数会调用getenv寻找env对应的LazyLLM环境变量，若找到则写入config。如果env为一个字典，函数将尝试调用getenv寻找字典中的key对于的环境变量并转换为bool型。
若转换完成的bool值是True，则将字典中当前的key对应的值写入config。

参数:
    name (str): 配置项名称
    type (type): 该配置的类型
    default (可选): 若无法获取到任何值，该配置的默认值
    env (可选): 不包含前缀的环境变量名称，或者一个字典，其中的key是不包含前缀的环境变量名称，value是要加入配置的值。

.. function:: Config.get_all_configs() -> None
获取config中所有的配置

参数：
    无
''')


add_english_doc('Config', '''\
Config is a configuration class provided by LazyLLM, which loads configurations of LazyLLM framework from config files,
environment variables, or specify them explicitly. it can export all configuration items as well.
The Config module automatically generates an object named 'config' containing all configurations.

.. function:: Config.done() -> None
Check if any configuration items in the config.json file that is not loaded by the add method.

Arguments:
    None.

.. function:: Config.getenv(name, type, default): -> str
Get value of LazyLLM-related environment variables.

Arguments:
    name (str): The name of the environment variable （without the prefix）, case-insensitive. The function obtains value
    from environment variable by concatenating the prefix and this name, with all uppercase letters.
    type (type): Specifies the type of the configuration, for example, str. For boolean types, the function will
    convert inputs ‘TRUE’, ‘True’, 1, ‘ON’, and ‘1’ to True.
    default (optional): If the value of the environment variable cannot be obtained, this value is returned.

.. function:: Config.add(name, type, default, env):: -> Config
Loads value into LazyLLM configuration item. The function first attempts to find the value with the given name from the
dict loaded from config.json. If found, it removes the key from the dict and saves the value to the config.
If 'env' is a string, the function calls getenv to look for the corresponding LazyLLM environment variable, and if
it's found, writes it to the config. If 'env' is a dictionary, the function attempts to call getenv to find the
environment variables corresponding to the keys in the dict and convert them to boolean type.
If the converted boolean value is True, the value corresponding to the current key in the dict is written to the config.

Arguments:
    name (str): The name of the configuration item
    type (type): The type of the configuration
    default (optional): The default value of the configuration if no value can be obtained
    env (optional): The name of the environment variable without the prefix, or a dictionary where the keys are the
    names of the environment variables(without the prefix), and the values are what to be added to the configuration.

.. function:: Config.get_all_configs() -> None
Get all configurations from the config.

Arguments:
    None.
''')

add_example('Config', '''\
>>> import lazyllm
>>> from lazyllm.configs import config
>>> config['launcher']
'empty'
>>> config.get_all_configs()
{'home': '~/.lazyllm/', 'mode': <Mode.Normal: (1,)>, 'repr_ml': False, 'rag_store': 'None', 'redis_url': 'None', 'gpu_type': 'A100', 'debug': False, 'log_name': 'lazyllm', 'log_level': 'INFO', 'log_format': '{process}: <green>{time:YYYY-MM-DD HH:mm:ss}</green> {extra[name]} <level>{level}</level>: ({name}) <cyan>{message}</cyan>', 'log_dir': '~/.lazyllm', 'log_file_level': 'ERROR', 'log_file_size': '4 MB', 'log_file_retention': '7 days', 'log_file_mode': 'merge', 'launcher': 'empty', 'partition': 'your_part', 'sco.workspace': 'your_workspace', 'model_source': 'modelscope', 'model_cache_dir': '/mnt/cache/wangzhihong/.lazyllm/model', 'model_path': '', 'model_source_token': '', 'redis_recheck_delay': 5, 'openai_api_key': '', 'moonshotai_api_key': '', 'glm_api_key': '', 'qwen_api_key': '', 'sensenova_ak': '', 'sensenova_sk': '', 'doubao_api_key': '', 'language': 'CHINESE'}
''')
