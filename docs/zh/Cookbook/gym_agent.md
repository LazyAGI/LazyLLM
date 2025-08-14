# 基于LazyLLM的模拟环境交互教程

本教程将展示如何使用 **LazyLLM** 结合 **Gymnasium Blackjack-v1** 环境，实现一个由 LLM 控制的自动玩家系统。你将看到如何：

- 使用 `ReactAgent` 控制环境
- 注册环境操作工具
- 输出每一步的观察、奖励、动作信息

------

## 项目依赖

安装 `lazyllm`：

```bash
pip install lazyllm
```

导入相关包：

```python
import gymnasium as gym

from typing import Literal, Any, Dict

import lazyllm
from lazyllm.tools import fc_register, ReactAgent
```


------

## 创建 Blackjack 环境

```python
# 创建 Blackjack 环境
env = gym.make("Blackjack-v1")
obs, info = env.reset()

# 全局状态变量
current_obs = obs
total_return = 0.0
terminated = False
truncated = False
```

------

## 注册环境操作工具

### 1. 重置环境

```python
@fc_register("tool")
def env_reset() -> Dict[str, Any]:
    """
    Reset the environment and return the initial observation.

    Returns:
        Dict[str, Any]: The initial observation and other info.
    """
    global current_obs, total_return, terminated, truncated
    obs, info = env.reset()
    current_obs = obs
    total_return = 0.0
    terminated = False
    truncated = False
    return {
        "observation": obs,
        "reward": 0.0,
        "termination": terminated,
        "truncation": truncated,
        "return": total_return
    }
```

### 2. 执行动作

```python
@fc_register("tool")
def env_step(action: int) -> Dict[str, Any]:
    """
    Take an action in the environment.

    Args:
        action (int): The action to take.

    Returns:
        Dict[str, Any]: The new observation, reward, termination, truncation, and return.
    """
    global current_obs, total_return, terminated, truncated
    obs, reward, terminated, truncated, info = env.step(action)
    current_obs = obs
    total_return += reward
    return {
        "observation": obs,
        "reward": float(reward),
        "termination": terminated,
        "truncation": truncated,
        "return": float(total_return)
    }
```

### 3. 随机动作采样

```python
@fc_register("tool")
def sample_random_action() -> int:
    """
    Sample a random action from the environment's action space.

    Returns:
        int: The sampled action.
    """
    return env.action_space.sample()
```

------

## 定义 LLM 与系统提示

```python
# 定义在线 LLM 模块
llm = lazyllm.OnlineChatModule()

# 系统提示
system_prompt = """
You are controlling a gymnasium environment. 
You can use env_reset() to start, env_step(action) to take a step, and sample_random_action() to choose a random action.
Always print:
Observation: ...
Reward: ...
Termination: ...
Truncation: ...
Return: ...
Action: ...
"""
```

------

## 创建 Agent

```python
agent = ReactAgent(
    llm,
    ["env_reset", "env_step", "sample_random_action"],
    prompt=system_prompt
)
```

------

## 主函数示例

```python
if __name__ == "__main__":
    # 重置环境
    obs = agent("Please reset the environment first.")
    print(obs)

    # 玩几步游戏
    for _ in range(5):
        # 随机动作
        action = agent("Please choose a random action and take a step.")
        print(action)
```

------

## 输出示例

```text
Observation: [14, 10, 0]
Reward: 0.0
Termination: false
Truncation: false
Return: 0.0

Observation: (20, 7, False)
Reward: 0.0
Termination: False
Truncation: False
Return: 0.0

Observation: (21, 7, False)
Reward: 1.0
Termination: True
Truncation: False
Return: 1.0
```

> 每一步 LLM 自动选择动作，并打印当前观测、奖励、终止状态以及累计返回。
