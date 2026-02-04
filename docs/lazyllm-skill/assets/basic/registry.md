# AutoRegistry 使用示例

## 基于类的自动注册

### 基础实现类注册

```python
from lazyllm.online.base import LazyLLMOnlineChatModuleBase

class MyChatModel(LazyLLMOnlineChatModuleBase):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        return f"[{self.model_name}] {prompt}"
```
注册后路径: lazyllm.online.chat.mychatmodel

### 显式指定 registry key

```python
from lazyllm.online.base import LazyLLMOnlineChatModuleBase

class MyChatModel(LazyLLMOnlineChatModuleBase):
    __lazyllm_registry_key__ = "mychat"

    def chat(self, prompt: str) -> str:
        return prompt
```
访问方式: lazyllm.online.chat.mychat(...)

### 禁止中间抽象类注册

```python
from lazyllm.online.base import LazyLLMOnlineChatModuleBase

class BaseChat(LazyLLMOnlineChatModuleBase):
    __lazyllm_registry_disable__ = True

    def format_prompt(self, p: str) -> str:
        return f"<chat>{p}</chat>"

class FancyChat(BaseChat):
    def chat(self, prompt: str):
        return self.format_prompt(prompt)
```
此时BaseChat不会注册，FancyChat会注册

### 多级继承注册示例

```python
class ProviderBase(LazyLLMOnlineChatModuleBase):
    __lazyllm_registry_disable__ = True

class OpenAIChat(ProviderBase):
    pass

class AzureChat(ProviderBase):
    pass
```
访问方式: lazyllm.online.chat.openaichat(...), lazyllm.online.chat.azurechat(...)

## 基于函数的注册（Register 装饰器）

### 注册为 tool

```python
from lazyllm.common import Register

@Register("tool")
def word_count(text: str) -> int:
    return len(text.split())
```
访问方式: lazyllm.tool.word_count("hello world")

### 注册为 dataset / parser 等分组

```python
@Register("dataset")
def load_demo_dataset():
    return ["a", "b", "c"]
```
访问方式: lazyllm.dataset.load_demo_dataset()

## 已注册能力的访问方式

### 通过 key 访问

```python
lazyllm.online.chat.mychat(...)
```

### 通过类名访问

```python
lazyllm.online.chat.MyChatModel(...)
```

### 默认实现访问

```python
lazyllm.embed(...)
```

## 常见错误示例（反例）

该部分内容均是禁止的操作行为，必须避免以下操作

### 手动维护registry

```python
REGISTRY = {}
REGISTRY["x"] = MyClass
```

### 直接实例化具体实现类作为主要入口

```python
model = MyChatModel()
```

应该修改为:

```python
model = lazyllm.online.chat.mychatmodel()
```

或者修改为:

```python
from lazyllm.online.chat import mychatmodel

model = mychatmodel()
```

### 抽象类忘记禁止注册

```python
class BaseX(LazyBase):
    pass
```

应该修改为:

```python
class BaseX(LazyBase):
    __lazyllm_registry_disable__ = True
```

## 最小可运行示例

```python
# my_chat.py
from lazyllm.online.base import LazyLLMOnlineChatModuleBase

class SimpleChat(LazyLLMOnlineChatModuleBase):
    def chat(self, prompt):
        return f"echo: {prompt}"
```

```python
# main.py
import lazyllm
import my_chat   # 触发注册

chat = lazyllm.online.chat.simplechat()
print(chat.chat("hi"))
```
