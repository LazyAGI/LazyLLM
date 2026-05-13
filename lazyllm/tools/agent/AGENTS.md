# lazyllm/tools/agent AGENTS.md

This directory implements LazyLLM's **Agent system**, including multiple Agent types, tool registration mechanisms, and the FunctionCall execution flow.

Reference docs: [`lazyllm/docs/module.py`](../../docs/module.py) (API docs for FunctionCall, ReactAgent, etc.)

## Mandatory Pre-reading

Before modifying this directory, you must read:
- `lazyllm/AGENTS.md` (global conventions)
- `lazyllm/tools/AGENTS.md` (lazy-loading mechanism)
- `lazyllm/module/AGENTS.md` (ModuleBase — Agent inherits from it)
- `lazyllm/flow/AGENTS.md` (Flow orchestration — Agent uses Pipeline + Loop internally)
- `lazyllm/tools/agent/base.py` (`LazyLLMAgentBase`)
- `lazyllm/tools/agent/functionCall.py` (FunctionCall core execution flow)

---

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `base.py` | `LazyLLMAgentBase` (base class for all Agents) |
| `functionCall.py` | `FunctionCall` (single-turn tool call execution unit), `FunctionCallAgent` |
| `reactAgent.py` | `ReactAgent` (ReAct loop Agent) |
| `planAndSolveAgent.py` | `PlanAndSolveAgent` (plan + execute Agent) |
| `rewooAgent.py` | `ReWOOAgent` (blueprint + evidence + answer Agent) |
| `toolsManager.py` | `ToolManager`, `ModuleTool`, `register` (tool registration) |
| `skill_manager.py` | `SkillManager` (persistent skill management) |
| `skill_hub.py` | `install_skill` (skill installation) |

---

## Agent Type Comparison

| Agent | Working method | Use case |
|-------|---------------|----------|
| `ReactAgent` | Reason→Act→Observe loop until a final answer is reached | General tasks requiring multi-step reasoning and tool calls |
| `PlanAndSolveAgent` | Planner decomposes subtasks; Solver executes the plan | Complex tasks that need planning before execution |
| `ReWOOAgent` | Planner generates a blueprint; Worker fills in evidence; Solver gives the answer | Tasks that need to collect information in parallel |
| `FunctionCallAgent` | Directly selects and calls a tool (deprecated; use `ReactAgent` instead) | Simple tool calls |

---

## FunctionCall Execution Flow

`FunctionCall` is the core execution unit for all Agents, implementing a single round of "LLM reasoning + tool call":

```
input
  ↓
_build_history (builds history messages, injected into locals['_lazyllm_agent']['workspace'])
  ↓
LLM (reasoning; outputs content or tool_calls)
  ↓
_post_action (parses output)
  ├── has tool_calls → calls ToolManager to execute tools → returns dict (continue loop)
  └── no tool_calls → returns str (triggers Loop stop condition)
```

### Loop Stop Condition

`ReactAgent` wraps `FunctionCall` in a `Loop`:

```python
agent_loop = loop(
    FunctionCall(llm=self._llm, _tool_manager=self._tools_manager, ...),
    stop_condition=lambda x: isinstance(x, str),  # stops when a string is returned
    count=self._max_retries,
)
```

- Returns `dict` (contains `tool_calls`) → continue loop
- Returns `str` (final answer) → stop loop

### History Message Management

`FunctionCall` stores conversation history in `locals['_lazyllm_agent']['workspace']` (not in instance variables):

```python
workspace = locals['_lazyllm_agent']['workspace']
workspace.setdefault('history', [])
workspace['history'].append({'role': 'user', 'content': input})
```

This ensures conversation history does not leak across users under concurrent requests.

---

## Tool Registration

### Method 1: Inherit `ModuleTool` (recommended for complex tools)

```python
from lazyllm.tools.agent.toolsManager import ModuleTool

class MySearchTool(ModuleTool):
    def apply(self, query: str, max_results: int = 5) -> str:
        '''Search for information on the web.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of results to return.

        Returns:
            str: Search results as formatted text.
        '''
        return do_search(query, max_results)
```

### Method 2: `@fc_register` decorator (for simple tools)

```python
import lazyllm

@lazyllm.tools.fc_register('tool')
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        str: The result of the evaluation.
    '''
    return str(eval(expression))
```

### Method 3: Pass a Callable directly to `ToolManager`

```python
def my_func(query: str) -> str:
    '''My tool description.

    Args:
        query (str): Input query.
    '''
    return process(query)

agent = ReactAgent(llm, tools=[my_func])
# ToolManager internally registers it in a temporary group, deleted after use
```

### docstring Format Requirements

A tool's docstring is used to auto-generate the OpenAI function calling schema. **The following format must be followed:**

```python
def my_tool(param1: str, param2: int = 5) -> str:
    '''One-line description of what the tool does.

    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2. Default is 5.

    Returns:
        str: Description of the return value.
    '''
```

- First line: short description of the tool (`short_description`, used for the `description` field)
- `Args:` block: type and description for each parameter (used for `parameters.properties`)
- Type annotations must be complete (used for `parameters.properties.type`)
- Tools without a docstring or with an incorrect format **cannot be called correctly by the LLM**

---

## ToolManager

`ToolManager` manages the tool collection and is responsible for:
1. Registering tools in a temporary group (`tmp_tool`)
2. Generating tool descriptions (`tools_description`) for the LLM
3. Executing tool calls (`_execute_tool`)

```python
tool_manager = ToolManager(
    tools=['calculator', 'search', my_custom_func],
    return_trace=False,
    sandbox=None,  # code execution sandbox
)

# Get tool descriptions (OpenAI function calling format)
descriptions = tool_manager.tools_description

# Execute a tool call
result = tool_manager._execute_tool('calculator', {'expression': '1+1'})
```

---

## LazyLLMAgentBase

Base class for all Agents, providing:
- `_llm`: LLM module
- `_tools`: tool list
- `_tools_manager`: ToolManager instance
- `_memory`: memory module (optional)
- `_skill_manager`: skill manager (optional)
- `_sandbox`: code execution sandbox (default `'auto'`)
- `_max_retries`: maximum number of retries

### Adding a New Agent Type

1. Inherit `LazyLLMAgentBase`
2. Call `super().__init__(...)` in `__init__`
3. Implement `build_agent()` to construct the internal Flow (typically `Loop(FunctionCall(...), ...)`)
4. Implement `forward(self, query, ...)` to call the internal Flow
5. Refer to `ReactAgent` for alignment

```python
class MyAgent(LazyLLMAgentBase):
    def __init__(self, llm, tools, **kwargs):
        super().__init__(llm=llm, tools=tools, **kwargs)
        self.build_agent()

    @once_wrapper
    def build_agent(self):
        self._agent = loop(
            FunctionCall(llm=self._llm, _tool_manager=self._tools_manager),
            stop_condition=lambda x: isinstance(x, str),
            count=self._max_retries,
        )

    def forward(self, query: str, **kwargs) -> str:
        self.build_agent()
        return self._agent(query)
```

---

## Skill System

A Skill is a reusable, persistable collection of tools. Each skill is a directory containing a `SKILL.md` file.

### Skill Directory Structure

```
my_skill/
├── SKILL.md          # Skill description (YAML frontmatter + Markdown body)
├── main.py           # Skill main script (optional)
└── reference/        # Reference files (optional)
    └── example.txt
```

`SKILL.md` YAML frontmatter format:

```yaml
---
name: web_search
description: Search the web for information using Bing or Google
---

## Usage
This skill searches the web and returns relevant results.

## Parameters
- query: The search query string
```

### SkillManager

`SkillManager` scans a local or cloud FS directory and manages the skill library:

```python
from lazyllm.tools.agent import SkillManager

# Local skill library
skill_manager = SkillManager(dir='./skills')

# Cloud skill library (Feishu)
from lazyllm.tools.fs import FeishuFS
skill_manager = SkillManager(dir='/skills', fs=FeishuFS(token='xxx'))

# Load specific skills only
skill_manager = SkillManager(dir='./skills', skills=['web_search', 'code_exec'])
```

Key methods of `SkillManager`:

| Method | Description |
|--------|-------------|
| `list_skill()` | Lists all available skills (returns `{name: description}` dict) |
| `get_skill(name)` | Reads the full `SKILL.md` content for a skill |
| `read_reference(name, rel_path)` | Reads a reference file inside the skill directory |
| `run_script(name, rel_path)` | Executes a script (Python/Shell) inside the skill directory |
| `get_skill_tools()` | Returns a list of tool functions that can be injected into an Agent |
| `wrap_input(input, task)` | Injects the skill list into the Agent's input |

### Using Skills in an Agent

```python
agent = ReactAgent(
    llm=OnlineModule('qwen-plus'),
    tools=[calculator, web_search],
    skills=True,                    # enable skills (loads from default directory)
    skills_dir='./my_skills',       # custom skill directory
)

# Or specify particular skills
agent = ReactAgent(
    llm=OnlineModule('qwen-plus'),
    tools=[],
    skills=['web_search', 'code_exec'],
)
```

### Installing Skills

```python
from lazyllm.tools.agent import install_skill

# Install from agentskillhub.dev
install_skill('agentskillhub:username/skill-slug')

# Install from GitHub
install_skill('github:owner/repo/path/to/skill')
```

After installation, skills are saved to a local directory and can be loaded via `SkillManager`.

---

## Prohibited Patterns

- **Never** store conversation history in Agent instance variables (use `locals['_lazyllm_agent']['workspace']`)
- **Never** omit docstrings or type annotations from tool functions (schema generation depends on them)
- **Never** access `globals` directly inside a tool function (tools should be stateless pure functions)
- **Never** implement tool call logic outside `FunctionCall` (reuse `FunctionCall` as the execution unit)
- **Never** override `__call__` (inherit `ModuleBase`'s `__call__`; only implement `forward`)
