# Simulated Environment: Gymnasium

This tutorial demonstrates how to use **LazyLLM** together with **Gymnasium Blackjack-v1** to implement an automated player system controlled by an LLM. You will learn how to:

* Control the environment using `ReactAgent`
* Register environment operation tools
* Print observations, rewards, and actions at each step

---

## Project Dependencies

Install `lazyllm`:

```bash
pip install lazyllm
```

Import necessary packages:

```python
import gymnasium as gym
from typing import Literal, Any, Dict
import lazyllm
from lazyllm.tools import fc_register, ReactAgent
```

---

## Create the Blackjack Environment

```python
# Create the Blackjack environment
env = gym.make("Blackjack-v1")
obs, info = env.reset()

# Global state variables
current_obs = obs
total_return = 0.0
terminated = False
truncated = False
```

---

## Register Environment Tools

### 1. Reset Environment

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

### 2. Take an Action

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

### 3. Sample a Random Action

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

---

## Define LLM and System Prompt

```python
# Define the online LLM module
llm = lazyllm.OnlineChatModule()

# System prompt
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

---

## Create the Agent

```python
agent = ReactAgent(
    llm,
    ["env_reset", "env_step", "sample_random_action"],
    prompt=system_prompt
)
```

---

## Example Main Function

```python
if __name__ == "__main__":
    # Reset the environment
    obs = agent("Please reset the environment first.")
    print(obs)

    # Play a few steps
    for _ in range(5):
        # Take a random action
        action = agent("Please choose a random action and take a step.")
        print(action)
```

---

## Example Output

```text
Observation: (14, 10, False)
Reward: 0.0
Termination: False
Truncation: False
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

> At each step, the LLM automatically selects an action and prints the current observation, reward, termination status, and cumulative return.
