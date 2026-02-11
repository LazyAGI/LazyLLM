from ..base_data import data_register
from lazyllm.components.formatter import JsonFormatter
import json

ToolUseOps = data_register.new_group('tool_use_ops')

class ScenarioExtractor(ToolUseOps):
    def __init__(self, model=None, input_key='content', output_key='scenario', system_prompt=None, **kwargs):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            '你是一个对话场景分析助手。你的任务是从对话内容中提取可用于数据生成的场景信息。\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "scene": "一句话场景描述",\n'
            '  "domain": "领域/主题",\n'
            '  "user_profile": "用户角色/背景（可为空）",\n'
            '  "assistant_goal": "助手应完成的目标",\n'
            '  "constraints": ["约束1","约束2"],\n'
            '  "key_entities": ["关键实体1","关键实体2"]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            data[self.output_key] = None
            return data
        instruction = f'对话内容如下：\n{content}\n\n请提取场景信息并输出 JSON。'
        parsed = self.model(instruction)
        data[self.output_key] = parsed if parsed is not None else ''
        return data


class ScenarioExpander(ToolUseOps):
    def __init__(
        self, model=None, input_key='scenario', output_key='expanded_scenarios', n=3, system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.n = n
        sys_prompt = system_prompt or (
            '你是一个场景扩展助手。你的任务是基于给定的原始场景，生成多个可替代的新场景，语义相关但细节不同。\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "scenarios": [\n'
            '    {"scene": "...", "domain": "...", "assistant_goal": "...", "constraints": ["..."], '
            '"key_entities": ["..."]}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        base = data.get(self.input_key, None)
        if base is None or base == '':
            data[self.output_key] = []
            return data
        base_text = json.dumps(base, ensure_ascii=False) if not isinstance(base, str) else base
        instruction = f'原始场景：\n{base_text}\n\n请生成 {self.n} 个替代场景并输出 JSON。'
        parsed = self.model(instruction)
        scenarios = parsed.get('scenarios') if isinstance(parsed, dict) else None
        data[self.output_key] = scenarios if isinstance(scenarios, list) else (parsed if parsed else [])
        return data


class AtomTaskGenerator(ToolUseOps):
    def __init__(
        self, model=None, input_key='scenario', output_key='atomic_tasks', n=5, system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.n = n
        sys_prompt = system_prompt or (
            '你是一个任务分解助手。你的任务是根据给定场景，生成一组可执行的原子任务（粒度小、单目标）。\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "tasks": [\n'
            '    {"task": "任务描述", "input": "输入（可为空）", "output": "输出（可为空）", "constraints": ["..."]}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        scenario = data.get(self.input_key, None)
        if scenario is None or scenario == '':
            data[self.output_key] = []
            return data
        scenario_text = json.dumps(scenario, ensure_ascii=False) if not isinstance(scenario, str) else scenario
        instruction = f'场景：\n{scenario_text}\n\n请生成不超过 {self.n} 个原子任务并输出 JSON。'
        parsed = self.model(instruction)
        tasks = parsed.get('tasks') if isinstance(parsed, dict) else None
        data[self.output_key] = tasks if isinstance(tasks, list) else (parsed if parsed else [])
        return data


class SequentialTaskGenerator(ToolUseOps):
    def __init__(
        self, model=None, input_key='atomic_tasks', output_key='sequential_tasks', system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            '你是一个任务编排助手。你的任务是根据原子任务集合，生成：\n'
            '1) 每个任务的后继任务（next_task）\n'
            '2) 由两者组合形成的组合任务（composed_task）\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "items": [\n'
            '    {"task": "...", "next_task": "...", "composed_task": "..."}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        tasks = data.get(self.input_key, None)
        if not tasks:
            data[self.output_key] = []
            return data
        tasks_text = json.dumps(tasks, ensure_ascii=False) if not isinstance(tasks, str) else tasks
        instruction = f'原子任务列表：\n{tasks_text}\n\n请生成后继与组合任务并输出 JSON。'
        parsed = self.model(instruction)
        items = parsed.get('items') if isinstance(parsed, dict) else None
        data[self.output_key] = items if isinstance(items, list) else (parsed if parsed else [])
        return data


class ParaSeqTaskGenerator(ToolUseOps):
    def __init__(
        self, model=None, input_key='atomic_tasks', output_key='para_seq_tasks', system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            '你是一个任务组合生成助手。你的任务是基于原子任务生成三类任务：\n'
            '1) 并行任务（parallel_tasks）：可以同时进行的任务组合\n'
            '2) 后继任务（sequential_tasks）：有明确先后依赖的任务组合\n'
            '3) 组合任务（hybrid_tasks）：包含并行与先后依赖的混合组合\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "parallel_tasks": ["..."],\n'
            '  "sequential_tasks": ["..."],\n'
            '  "hybrid_tasks": ["..."]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        tasks = data.get(self.input_key, None)
        if not tasks:
            data[self.output_key] = {'parallel_tasks': [], 'sequential_tasks': [], 'hybrid_tasks': []}
            return data
        tasks_text = json.dumps(tasks, ensure_ascii=False) if not isinstance(tasks, str) else tasks
        instruction = f'原子任务列表：\n{tasks_text}\n\n请生成三类任务并输出 JSON。'
        parsed = self.model(instruction)
        default_val = {'parallel_tasks': [], 'sequential_tasks': [], 'hybrid_tasks': []}
        data[self.output_key] = parsed if parsed is not None else default_val
        return data


class CompositionTaskFilter(ToolUseOps):
    def __init__(
        self, model=None, composition_key='composition_tasks', subtask_key='atomic_tasks',
        output_key='filtered_composition_tasks', system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.composition_key = composition_key
        self.subtask_key = subtask_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            '你是一个任务可运行性评审助手。你需要判断组合任务是否具备可行性与完备性：\n'
            '- 可行性：子任务是否能支撑组合任务目标\n'
            '- 完备性：是否缺少关键步骤或前置条件\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "items": [\n'
            '    {"composed_task": "...", "is_valid": true, "reason": "..."}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        composition_tasks = data.get(self.composition_key, None)
        subtasks = data.get(self.subtask_key, None)
        if not composition_tasks:
            data[self.output_key] = []
            return data
        composition_text = json.dumps(composition_tasks, ensure_ascii=False) if not isinstance(composition_tasks, str) \
            else composition_tasks
        subtasks_text = json.dumps(subtasks, ensure_ascii=False) if subtasks is not None \
            and not isinstance(subtasks, str) else (subtasks or '')
        instruction = (
            f'组合任务：\n{composition_text}\n\n'
            f'子任务（可选）：\n{subtasks_text}\n\n'
            '请逐条判断并输出 JSON。'
        )
        parsed = self.model(instruction)
        items = parsed.get('items') if isinstance(parsed, dict) else None
        valid = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get('is_valid') is True and it.get('composed_task'):
                    valid.append(it.get('composed_task'))
        data[self.output_key] = valid if valid else (
            items if isinstance(items, list) else (parsed if parsed else [])
        )
        return data


class FunctionGenerator(ToolUseOps):
    def __init__(
        self, model=None, task_key='composition_task', subtask_key='atomic_tasks',
        output_key='functions', system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.task_key = task_key
        self.subtask_key = subtask_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            '你是一个函数设计助手。给定组合任务及其子任务，请生成一组函数规格，便于后续工具调用。\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "functions": [\n'
            '    {"name": "function_name", "description": "...", '
            '"args": [{"name":"...","type":"...","description":"..."}], '
            '"returns": {"type":"...","description":"..."}}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        task = data.get(self.task_key, None)
        subtasks = data.get(self.subtask_key, None)
        if task is None or task == '':
            data[self.output_key] = []
            return data
        task_text = json.dumps(task, ensure_ascii=False) if not isinstance(task, str) else task
        subtasks_text = json.dumps(subtasks, ensure_ascii=False) if subtasks is not None \
            and not isinstance(subtasks, str) else (subtasks or '')
        instruction = (
            f'组合任务：\n{task_text}\n\n'
            f'子任务（可选）：\n{subtasks_text}\n\n'
            '请生成函数列表并输出 JSON。'
        )
        parsed = self.model(instruction)
        funcs = parsed.get('functions') if isinstance(parsed, dict) else None
        data[self.output_key] = funcs if isinstance(funcs, list) else (parsed if parsed else [])
        return data


class MultiTurnConversationGenerator(ToolUseOps):
    def __init__(
        self, model=None, task_key='composition_task', functions_key='functions',
        output_key='conversation', n_turns=6, system_prompt=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.task_key = task_key
        self.functions_key = functions_key
        self.output_key = output_key
        self.n_turns = n_turns
        sys_prompt = system_prompt or (
            '你是一个多轮对话数据生成助手。你需要根据组合任务与可用函数，模拟一段多轮对话。\n'
            '对话由 User/Assistant/Tool 三种角色组成：\n'
            '- User 提出需求与补充信息\n'
            '- Assistant 规划并在适当时机调用 Tool\n'
            '- Tool 返回函数执行结果\n'
            '只输出 JSON，不要输出任何额外文本。\n'
            'JSON 结构：\n'
            '{\n'
            '  "messages": [\n'
            '    {"role":"user","content":"..."},\n'
            '    {"role":"assistant","content":"..."},\n'
            '    {"role":"tool","name":"function_name","content":"..."}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        task = data.get(self.task_key, None)
        functions = data.get(self.functions_key, None)
        if task is None or task == '':
            data[self.output_key] = []
            return data
        task_text = json.dumps(task, ensure_ascii=False) if not isinstance(task, str) else task
        functions_text = json.dumps(functions, ensure_ascii=False) if functions is not None \
            and not isinstance(functions, str) else (functions or '')
        instruction = (
            f'组合任务：\n{task_text}\n\n'
            f'函数列表：\n{functions_text}\n\n'
            f'请生成约 {self.n_turns} 轮对话的 messages 并输出 JSON。'
        )
        parsed = self.model(instruction)
        data[self.output_key] = parsed if parsed is not None else []
        return data
