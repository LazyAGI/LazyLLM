# Environment Interaction with LazyLLM

In many LLM agent applications, the agent needs to interact with real-world environments such as the internet, databases, or code execution (REPL). However, for easier development and testing, we can also create simulated environments—such as text-based adventure games—where the agent can make decisions and engage in interactions. This tutorial demonstrates how to use [LazyLLM](https://github.com/LazyAGI/LazyLLM) in conjunction with the **Gymnasium Blackjack-v1** environment to build a simple text-based interaction loop, enabling seamless dialogue and response cycles between the agent and the environment.
!!! abstract "By the end of this section, you will learn the following key features of LazyLLM"
    - How to control an environment using [ReactAgent][lazyllm.tools.agent.ReactAgent]
    - How to register environment operation tools via @fc_register

------
## Feature Overview
This is a game of Blackjack, where the player and dealer compete to get as close to 21 as possible without going over. The player can choose to “hit” (draw another card) or “stick” (stop drawing). The dealer follows a fixed rule: it must hit if its total is below 17 and must stick if it’s 17 or higher. The Ace is special—it can count as either 1 or 11 points; the system automatically uses 11 unless that would cause a bust, in which case it drops to 1. If the hand contains an Ace counted as 11 points, it is called a "soft hand." If the Ace is counted as 1 point, it is called a "hard hand." If the player busts, they lose; if the dealer busts, the player wins; if neither busts, the higher total wins. In this setup, the Gymnasium environment acts as the dealer, while the Agent plays as the player, interacting with the environment by calling tools to decide when to hit or stick—learning to make smart decisions in the game of Blackjack. 

The specific functions used in the game are as follows:

* Automatically initialize the `Gymnasium Blackjack-v1` environment and manage global state (observation, return, termination status, etc.).
* Register three tool functions — `env_reset`, `env_step`, and `sample_random_action` — exposed as callable interfaces for LLMs via fc_register, enabling environment reset, action execution, and random action sampling.
* Construct a conversational intelligent agent using `ReactAgent` to enable natural language-driven interactive reasoning over the environment.
* Enhance robustness through the safe wrapper `safe_agent_call`, which automatically handles unstructured responses or exceptions.

## Design Concept

First, the system creates the Blackjack-v1 environment and initializes global state variables. Next, the system constructs a ReactAgent based on an LLM, providing it with a list of registered environment operation tools. Then, the interaction loop between the agent and the environment begins: in each step, the agent calls the appropriate tools (e.g., first sampling a random action, then executing a step) based on the current state and system instructions, and the system captures and parses the agent’s response to extract the commanded action.
![gym](../assets/gym.png)


## Code Implementation
### Project Dependencies

Install `lazyllm`：

```bash
pip install lazyllm
```

Import required packages:

```python
import gymnasium as gym
from typing import Any, Dict
import lazyllm
from lazyllm.tools import fc_register, ReactAgent
```

------

### Create the Blackjack Environment
`Blackjack-v1` is an environment in OpenAI Gym that simulates Blackjack, a classic card game of twenty-one. The player's goal is to get the total value of their cards as close to 21 as possible without exceeding it, and to win by comparing their point total with the dealer's by choosing to "hit" or "stand".
```python
# Create the Blackjack Environment
env = gym.make('Blackjack-v1')
obs, info = env.reset()
# Global state variables
current_obs = obs
total_return = 0.0
terminated = False
truncated = False

```

------

### Step-by-Step Explanation

#### Step1: Reset the Environment
The `env_reset` function resets Blackjack-v1 to its initial state. The `reset` method fetches the initial observation and info, updates global variables, and returns a dictionary containing the initial observation, reward, termination/truncation flags, and total return.
```python
@fc_register('tool')
def env_reset() -> Dict[str, Any]:
    '''
    Reset the environment and return the initial observation.
    Returns:
        Dict[str, Any]: The initial observation and other info.
    '''
    global current_obs, total_return, terminated, truncated
    obs, info = env.reset()
    current_obs = obs
    total_return = 0.0
    terminated = False
    truncated = False
    return {
        'observation': obs,
        'reward': 0.0,
        'termination': terminated,
        'truncation': truncated,
        'return': total_return
    }
```

#### Step2: Take an action
The `env_step` function executes an action. It first checks if the current round has ended; if so, it returns the current state and an error message. If the round is still active, it calls the underlying environment’s step method to perform the action.
```python
@fc_register('tool')
def env_step(action: int) -> Dict[str, Any]:
    '''
    Take an action in the environment.
    Args:
        action (int): The action to take.
    Returns:
        Dict[str, Any]: The new observation, reward, termination, truncation, and return.
    '''
    global current_obs, total_return, terminated, truncated
    if terminated or truncated:
        return {
            'observation': current_obs,
            'reward': 0.0,
            'termination': terminated,
            'truncation': truncated,
            'return': total_return,
            'error': 'Cannot step after episode termination'
        }

    obs, reward, terminated, truncated, info = env.step(action)
    current_obs = obs
    total_return += reward
    return {
        'observation': obs,
        'reward': float(reward),
        'termination': terminated,
        'truncation': truncated,
        'return': float(total_return)
    }

```

### Step3: Random action sampling
The `sample_random_action` function samples a random action from the Blackjack-v1 environment.
```python
@fc_register('tool')
def sample_random_action() -> Dict[str, Any]:
    '''
    Sample a random action from the environment's action space.
    Returns:
        Dict[str, Any]: The sampled action.
    '''
    action = env.action_space.sample()
    return {'action': int(action)}
```

------

#### Step4: Define the LLM and create the Agent
Build the agent by instantiating ReactAgent with llm and tools.
```python
llm = lazyllm.OnlineChatModule()

system_prompt = '''
You are controlling a gymnasium environment.
- Use env_reset() to start a new game.
- Use sample_random_action() to get an action, then env_step(action) to take it.
- **If termination is true, do NOT call env_step again.**
- Always output EXACTLY in this format:
Observation: [x, y, z]
Reward: x.x
Termination: true/false
Truncation: true/false
Return: x.x
Action: 0 or 1
'''
agent = ReactAgent(
    llm,
    ['env_reset', 'env_step', 'sample_random_action'],
    prompt=system_prompt
)

```

------

### View full code
<details>
<summary>Click to expand/collapse Python code</summary>

```python
import gymnasium as gym
from typing import Any, Dict
import lazyllm
from lazyllm.tools import fc_register, ReactAgent

env = gym.make('Blackjack-v1')
obs, info = env.reset()
current_obs = obs
total_return = 0.0
terminated = False
truncated = False


@fc_register('tool')
def env_reset() -> Dict[str, Any]:
    '''
    Reset the environment and return the initial observation.
    Returns:
        Dict[str, Any]: The initial observation and other info.
    '''
    global current_obs, total_return, terminated, truncated
    obs, info = env.reset()
    current_obs = obs
    total_return = 0.0
    terminated = False
    truncated = False
    return {
        'observation': obs,
        'reward': 0.0,
        'termination': terminated,
        'truncation': truncated,
        'return': total_return
    }


@fc_register('tool')
def env_step(action: int) -> Dict[str, Any]:
    '''
    Take an action in the environment.
    Args:
        action (int): The action to take.
    Returns:
        Dict[str, Any]: The new observation, reward, termination, truncation, and return.
    '''
    global current_obs, total_return, terminated, truncated
    if terminated or truncated:
        return {
            'observation': current_obs,
            'reward': 0.0,
            'termination': terminated,
            'truncation': truncated,
            'return': total_return,
            'error': 'Cannot step after episode termination'
        }

    obs, reward, terminated, truncated, info = env.step(action)
    current_obs = obs
    total_return += reward
    return {
        'observation': obs,
        'reward': float(reward),
        'termination': terminated,
        'truncation': truncated,
        'return': float(total_return)
    }


@fc_register('tool')
def sample_random_action() -> Dict[str, Any]:
    '''
    Sample a random action from the environment's action space.
    Returns:
        Dict[str, Any]: The sampled action.
    '''
    action = env.action_space.sample()
    return {'action': int(action)}


llm = lazyllm.OnlineChatModule()

system_prompt = '''
You are controlling a gymnasium environment.
- Use env_reset() to start a new game.
- Use sample_random_action() to get an action, then env_step(action) to take it.
- **If termination is true, do NOT call env_step again.**
- Always output EXACTLY in this format:
Observation: [x, y, z]
Reward: x.x
Termination: true/false
Truncation: true/false
Return: x.x
Action: 0 or 1

'''

agent = ReactAgent(
    llm,
    ['env_reset', 'env_step', 'sample_random_action'],
    prompt=system_prompt
)


def safe_agent_call(prompt: str):
    '''
    Safely invoke the ReactAgent, automatically retrying or skipping invalid responses.
    '''
    try:
        resp = agent(prompt)
        if isinstance(resp, str) and 'Observation' not in resp:
            print(f'⚠️ Unexpected response format: {resp}')
        return resp
    except Exception as e:
        print(f'⚠️ Agent call failed: {e}')
        env_reset()
        return {'error': str(e)}


if __name__ == '__main__':
    obs = safe_agent_call('Please reset the environment first.')
    print(obs)
    step_count = 0
    max_steps = 5
    while step_count < max_steps:
        if terminated or truncated:
            print('Game has ended. Stopping.')
            break

        action = safe_agent_call('Please choose a random action and take a step.')
        print(action)
        if isinstance(action, dict) and action.get('error'):
            print('⚠️ Encountered error, retrying...')
            continue

        step_count += 1

```
</details>
------

## Sample Output
The interaction logs from the Gymnasium Blackjack-v1 environment are as follows:

```text
Observation: [14, 10, 0] - The player's current hand total is 14, the dealer's visible card is 10, and no Ace is being used as 11 (0 indicates False).
Reward: 0.0 - No reward yet (game is still ongoing).
Termination: false - Game has not ended.
Truncation: false - Not truncated due to step limits.
Return: 0.0 - Cumulative reward is 0.

Observation: (20, 7, False) - After hitting again, the player's hand total becomes 20, the dealer's visible card is 7, and no soft Ace is present(The player's hand contains no Ace, or any Ace present is being counted as 1 point).
Reward: 0.0 - Game is still in progress.
Termination: False - Game has not ended.
Truncation: False - Game is proceeding normally.
Return: 0.0 - Cumulative reward is still 0.

Observation: (21, 7, False) - After one more hit, the player reaches exactly 21, while the dealer's visible card is 7.
Reward: 1.0 - Reward of +1 awarded for winning with 21.
Termination: True - Game ends (player wins).
Truncation: False - Termination is due to game outcome, not step limits.
Return: 1.0 - Final cumulative reward is 1.
```
## Conclusion
This tutorial demonstrates how to build an **Environment Interaction Agent** using LazyLLM, suitable for game strategy simulation, environment control experiments, and automated decision-making research. By combining `fc_register`, `OnlineChatModule`, and `ReactAgent`, you can rapidly construct an intelligent agent that understands natural language commands, dynamically invokes environment tools, and performs multi-turn reasoning — enabling a "language-as-an-interface" paradigm for human-agent collaboration.