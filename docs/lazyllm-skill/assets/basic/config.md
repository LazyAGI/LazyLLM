# Config使用示例

## 读取配置

### 通过 key 读取
```python
import lazyllm

mode = lazyllm.config["mode"]
print(mode)
```

### 带默认值读取
```python
gpu = lazyllm.config.get("gpu_type", "A100")
```

### 读取全部配置
```python
configs = lazyllm.config.get_all_configs()
print(configs)
```

## 设置与注册配置项

### 注册新的配置项
```python
import lazyllm

lazyllm.config.add(
    name="max_retry",
    default=3,
    type=int,
    description="Maximum retry times for network request"
)
```

### 在代码中设置配置值
```python
lazyllm.config["max_retry"] = 5
```

## 从环境变量读取配置

### 通过 getenv 读取
```python
timeout = lazyllm.config.getenv(
    "timeout",
    int,
    default=30
)
```

### 对应环境变量形式
```python
LAZYLLM_TIMEOUT=60
```

## 使用配置文件

### 默认配置文件路径
```python
~/.lazyllm/config.json
```

### 示例配置文件内容
```python
{
  "mode": "Debug",
  "gpu_type": "A100",
  "max_retry": 4
}
```

## 临时覆盖配置（temp）

### 单个配置项
```python
import lazyllm

with lazyllm.config.temp("gpu_type", "V100"):
    run_infer()

# 出了 with 作用域后自动恢复
```

### 多个配置项
```python
with lazyllm.config.temp(
    gpu_type="V100",
    mode="Debug"
):
    run_debug()
```

## 在模块中使用配置

### 在 init 中读取
```python
class MyModule:
    def __init__(self):
        import lazyllm
        self.max_retry = lazyllm.config.get("max_retry", 3)
```

### 在运行时读取
```python
def run():
    import lazyllm
    if lazyllm.config["mode"] == "Debug":
        enable_log()
```

## 常见错误示例（反例）

该部分内容均是禁止的操作行为，必须避免以下操作

### 直接使用 os.environ
```python
import os
timeout = os.environ["LAZYLLM_TIMEOUT"]
```

应修改为:
```python
timeout = lazyllm.config.getenv("timeout", int, 30)
```

### 硬编码全局变量
```python
GPU_TYPE = "A100"
```

应修改为:
```python
gpu = lazyllm.config["gpu_type"]
```

### 运行期修改配置文件
```python
open("~/.lazyllm/config.json", "w").write(...)
```

## 最小可运行示例

```python
import lazyllm

# 注册配置项
lazyllm.config.add(
    name="max_retry",
    default=3,
    type=int
)

# 读取
print(lazyllm.config["max_retry"])

# 临时覆盖
with lazyllm.config.temp("max_retry", 10):
    print(lazyllm.config["max_retry"])

print(lazyllm.config["max_retry"])
```
输出: 
3
10
3
