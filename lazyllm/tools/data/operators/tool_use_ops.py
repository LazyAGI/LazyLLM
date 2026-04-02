from ..base_data import data_register
from lazyllm.components.formatter import JsonFormatter
import json

ToolUseOps = data_register.new_group('tool_use_ops')


def _to_str(x, default=''):
    if x is None:
        return default
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


class ContextualBeacon(ToolUseOps):
    def __init__(self, model=None, input_key='content', output_key='scenario', system_prompt=None, **kwargs):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            'You are a dialogue scenario analysis assistant. Your task is to extract scenario information '
            'from conversation content for data generation.\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{\n'
            '  "scene": "One-sentence scenario description",\n'
            '  "domain": "Domain/topic",\n'
            '  "user_profile": "User role/background (optional)",\n'
            '  "assistant_goal": "Goal the assistant should achieve",\n'
            '  "constraints": ["constraint1","constraint2"],\n'
            '  "key_entities": ["entity1","entity2"]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        content = data.get(self.input_key, '')
        if not content:
            data[self.output_key] = None
            return data
        instruction = f'Conversation content:\n{content}\n\nExtract scenario information and output JSON.'
        parsed = self.model(instruction)
        data[self.output_key] = parsed if parsed is not None else ''
        return data


class ScenarioDiverger(ToolUseOps):
    def __init__(
        self, model=None, input_key='scenario', output_key='expanded_scenarios', n=3, system_prompt=None, **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.n = n
        sys_prompt = system_prompt or (
            'You are a scenario expansion assistant. Your task is to generate multiple alternative scenarios '
            'based on the given base scenario, semantically related but with different details.\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
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
        base = data.get(self.input_key)
        if not base and base != 0:
            data[self.output_key] = []
            return data
        base_text = _to_str(base)
        instruction = f'Base scenario:\n{base_text}\n\nGenerate {self.n} alternative scenarios and output JSON.'
        parsed = self.model(instruction)
        scenarios = parsed.get('scenarios') if isinstance(parsed, dict) else None
        data[self.output_key] = scenarios if isinstance(scenarios, list) else (parsed if parsed else [])
        return data


class DecompositionKernel(ToolUseOps):
    def __init__(
        self, model=None, input_key='scenario', output_key='atomic_tasks', n=5, system_prompt=None, **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.n = n
        sys_prompt = system_prompt or (
            'You are a task decomposition assistant. Your task is to generate a set of executable atomic tasks '
            '(fine-grained, single-goal) based on the given scenario.\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{\n'
            '  "tasks": [\n'
            '    {"task": "Task description", "input": "Input (optional)", "output": "Output (optional)", '
            '"constraints": ["..."]}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        scenario = data.get(self.input_key)
        if not scenario and scenario != 0:
            data[self.output_key] = []
            return data
        scenario_text = _to_str(scenario)
        instruction = f'Scenario:\n{scenario_text}\n\nGenerate up to {self.n} atomic tasks and output JSON.'
        parsed = self.model(instruction)
        tasks = parsed.get('tasks') if isinstance(parsed, dict) else None
        data[self.output_key] = tasks if isinstance(tasks, list) else (parsed if parsed else [])
        return data


class ChainedLogicAssembler(ToolUseOps):
    def __init__(
        self, model=None, input_key='atomic_tasks', output_key='sequential_tasks', system_prompt=None, **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            'You are a task orchestration assistant. Your task is to generate based on a set of atomic tasks:\n'
            '1) The successor task for each task (next_task)\n'
            '2) A composed task formed by combining the two (composed_task)\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{\n'
            '  "items": [\n'
            '    {"task": "...", "next_task": "...", "composed_task": "..."}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        tasks = data.get(self.input_key)
        if not tasks:
            data[self.output_key] = []
            return data
        tasks_text = _to_str(tasks)
        instruction = f'Atomic task list:\n{tasks_text}\n\nGenerate successor and composed tasks and output JSON.'
        parsed = self.model(instruction)
        items = parsed.get('items') if isinstance(parsed, dict) else None
        data[self.output_key] = items if isinstance(items, list) else (parsed if parsed else [])
        return data


class TopologyArchitect(ToolUseOps):
    def __init__(
        self, model=None, input_key='atomic_tasks', output_key='para_seq_tasks', system_prompt=None, **kwargs
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.input_key = input_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            'You are a task composition generation assistant. Your task is to generate three types of tasks '
            'based on atomic tasks:\n'
            '1) parallel_tasks: Task combinations that can be executed simultaneously\n'
            '2) sequential_tasks: Task combinations with clear sequential dependencies\n'
            '3) hybrid_tasks: Mixed combinations containing both parallel and sequential dependencies\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{\n'
            '  "parallel_tasks": ["..."],\n'
            '  "sequential_tasks": ["..."],\n'
            '  "hybrid_tasks": ["..."]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        tasks = data.get(self.input_key)
        if not tasks:
            data[self.output_key] = {'parallel_tasks': [], 'sequential_tasks': [], 'hybrid_tasks': []}
            return data
        tasks_text = _to_str(tasks)
        instruction = f'Atomic task list:\n{tasks_text}\n\nGenerate three types of tasks and output JSON.'
        parsed = self.model(instruction)
        default_val = {'parallel_tasks': [], 'sequential_tasks': [], 'hybrid_tasks': []}
        data[self.output_key] = parsed if parsed is not None else default_val
        return data


class ViabilitySieve(ToolUseOps):
    def __init__(
        self,
        model=None,
        input_composition_key='composition_tasks',
        input_atomic_key='atomic_tasks',
        output_key='filtered_composition_tasks',
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.composition_key = input_composition_key
        self.subtask_key = input_atomic_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            'You are a task feasibility review assistant. You need to evaluate whether composed tasks '
            'are feasible and complete:\n'
            '- Feasibility: Can subtasks support the composed task goal?\n'
            '- Completeness: Are critical steps or prerequisites missing?\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{\n'
            '  "items": [\n'
            '    {"composed_task": "...", "is_valid": true, "reason": "..."}\n'
            '  ]\n'
            '}\n'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def forward(self, data, **kwargs):
        assert isinstance(data, dict)
        composition_tasks = data.get(self.composition_key)
        subtasks = data.get(self.subtask_key)
        if not composition_tasks:
            data[self.output_key] = []
            return data
        composition_text = _to_str(composition_tasks)
        subtasks_text = _to_str(subtasks)
        instruction = (
            f'Composed tasks:\n{composition_text}\n\n'
            f'Subtasks (optional):\n{subtasks_text}\n\n'
            'Evaluate each and output JSON.'
        )
        parsed = self.model(instruction)
        items = parsed.get('items') if isinstance(parsed, dict) else None
        valid = [
            it['composed_task']
            for it in (items or [])
            if isinstance(it, dict) and it.get('is_valid') is True and it.get('composed_task')
        ]
        data[self.output_key] = valid
        return data


class ProtocolSpecifier(ToolUseOps):
    def __init__(
        self,
        model=None,
        input_composition_key='composition_task',
        input_atomic_key='atomic_tasks',
        output_key='functions',
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.task_key = input_composition_key
        self.subtask_key = input_atomic_key
        self.output_key = output_key
        sys_prompt = system_prompt or (
            'You are a function design assistant. Given a composed task and its subtasks, '
            'generate a set of function specifications for tool calling.\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
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
        task = data.get(self.task_key)
        subtasks = data.get(self.subtask_key)
        if isinstance(task, list):
            task = task[0] if task else None
        if not task and task != 0:
            data[self.output_key] = []
            return data
        task_text = _to_str(task)
        subtasks_text = _to_str(subtasks)
        instruction = (
            f'Composed task:\n{task_text}\n\n'
            f'Subtasks (optional):\n{subtasks_text}\n\n'
            'Generate function list and output JSON.'
        )
        parsed = self.model(instruction)
        funcs = parsed.get('functions') if isinstance(parsed, dict) else None
        data[self.output_key] = funcs if isinstance(funcs, list) else (parsed if parsed else [])
        return data


class DialogueSimulator(ToolUseOps):
    def __init__(
        self,
        model=None,
        input_composition_key='composition_task',
        input_functions_key='functions',
        output_key='conversation',
        n_turns=6,
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.task_key = input_composition_key
        self.functions_key = input_functions_key
        self.output_key = output_key
        self.n_turns = n_turns
        sys_prompt = system_prompt or (
            'You are a multi-turn dialogue data generation assistant. You need to simulate a multi-turn '
            'dialogue based on the composed task and available functions.\n'
            'The dialogue consists of three roles: User/Assistant/Tool:\n'
            '- User: Proposes requirements and supplementary information (in English)\n'
            '- Assistant: Plans and calls Tool when appropriate (responds in English)\n'
            '- Tool: Returns function execution results\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
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
        task = data.get(self.task_key)
        functions = data.get(self.functions_key)
        if isinstance(task, list):
            task = task[0] if task else None
        if not task and task != 0:
            data[self.output_key] = []
            return data
        task_text = _to_str(task)
        functions_text = _to_str(functions)
        instruction = (
            f'Composed task:\n{task_text}\n\n'
            f'Function list:\n{functions_text}\n\n'
            f'Generate approximately {self.n_turns} turns of dialogue messages and output JSON.'
        )
        parsed = self.model(instruction)
        data[self.output_key] = parsed if parsed is not None else []
        return data


class ToolUseToSFTFormatter(ToolUseOps):
    FORMAT_ALPACA = 'alpaca'
    FORMAT_CHATML = 'chatml'

    def __init__(self, format_type=FORMAT_CHATML, system_prompt=None, **kwargs):
        if format_type not in (self.FORMAT_ALPACA, self.FORMAT_CHATML):
            raise ValueError(f'Unknown format_type: {format_type!r}')
        super().__init__(**kwargs)
        self.format_type = format_type
        self.system_prompt = system_prompt or 'You are a helpful assistant that can use tools to help users.'

    def _format_functions(self, functions):
        if not functions:
            return '[]'
        return json.dumps(functions, ensure_ascii=False, indent=2)

    def _unpack_data(self, data):
        content = data.get('content', '')
        functions = data.get('functions', [])
        messages = data.get('conversation', {}).get('messages', [])
        return content, functions, messages

    def _convert_to_alpaca(self, data):
        content, functions, messages = self._unpack_data(data)
        tools_text = self._format_functions(functions)
        instruction = f'{self.system_prompt}\n\nAvailable tools:\n{tools_text}'

        output = ''
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                msg_content = msg.get('content', '')
                if msg_content:
                    output = msg_content
                    break

        return {
            'instruction': instruction,
            'input': content,
            'output': output
        }

    def _convert_to_chatml(self, data):
        content, functions, messages = self._unpack_data(data)
        tools_text = self._format_functions(functions)
        system_content = f'{self.system_prompt}\n\nAvailable tools:\n{tools_text}'
        output_messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': content},
        ]

        func_names = {func.get('name', '') for func in functions if func.get('name')}
        tool_call_id = 0
        for msg in messages:
            role = msg.get('role')
            msg_content = msg.get('content', '')

            if role == 'assistant':
                tool_calls = [
                    {
                        'id': f'call_{tool_call_id + i}',
                        'type': 'function',
                        'function': {'name': name, 'arguments': '{}'}
                    }
                    for i, name in enumerate(name for name in func_names if name in msg_content)
                ]
                if tool_calls:
                    tool_call_id += len(tool_calls)
                    output_messages.append({'role': 'assistant', 'content': None, 'tool_calls': tool_calls})
                else:
                    output_messages.append({'role': 'assistant', 'content': msg_content})

            elif role == 'tool':
                output_messages.append({
                    'role': 'tool',
                    'tool_call_id': f'call_{tool_call_id - 1}',
                    'content': msg_content
                })

        return {'messages': output_messages}

    def forward(self, data, **kwargs):
        if isinstance(data, list):
            return [r for item in data for r in [self.forward(item, **kwargs)] if r]

        assert isinstance(data, dict)

        if 'conversation' not in data or 'functions' not in data:
            return []

        if self.format_type == self.FORMAT_ALPACA:
            return self._convert_to_alpaca(data)
        return self._convert_to_chatml(data)


class ToolUseQualityFilter(ToolUseOps):
    def __init__(
        self,
        model=None,
        min_completeness_score=4,
        min_feasibility_score=4,
        system_prompt=None,
        **kwargs,
    ):
        super().__init__(_concurrency_mode='thread', **kwargs)
        self.min_completeness_score = min_completeness_score
        self.min_feasibility_score = min_feasibility_score
        sys_prompt = system_prompt or (
            'You are a data quality evaluation expert. Evaluate the quality of the assistant response '
            'based on the instruction and input.\n'
            'Scoring criteria (1-5 scale):\n'
            '- Completeness: Does the response fully address the task? Are all necessary steps covered?\n'
            '- Feasibility: Can the task be realistically accomplished? Is the response practical?\n'
            'Output only JSON, no extra text.\n'
            'JSON structure:\n'
            '{"completeness": 4, "feasibility": 3}'
        )
        self.model = model.share().prompt(sys_prompt).formatter(JsonFormatter())

    def _extract_eval_fields(self, data):
        formatted = data.get('formatted', {})
        if isinstance(formatted, dict) and formatted:
            instruction = formatted.get('instruction', '')
            input_data = formatted.get('input', '')
            output = formatted.get('output', '')
            if not output and 'messages' in formatted:
                messages = formatted['messages']
                # skip system message (index 0) to find the user instruction
                user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                instruction = user_msg.get('content', '') if user_msg else ''
                input_data = data.get('content', '')
                output = ''
                for msg in reversed(messages):
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        tool_calls = msg.get('tool_calls', [])
                        output = content if content else json.dumps(tool_calls, ensure_ascii=False)
                        break
        else:
            instruction = data.get('instruction', '')
            input_data = data.get('input', '')
            output = data.get('output', '')
        return instruction, input_data, output

    def forward(self, data, **kwargs):
        if isinstance(data, list):
            return [item for item in data if self.forward(item, **kwargs)]

        assert isinstance(data, dict)

        instruction, input_data, output = self._extract_eval_fields(data)
        if not output:
            return []

        eval_input = (
            f'Instruction:\n{instruction}\n\n'
            f'Input:\n{input_data}\n\n'
            f'Output:\n{output}\n\n'
            'Evaluate completeness and feasibility (1-5), output JSON.'
        )

        parsed = self.model(eval_input)
        if not isinstance(parsed, dict):
            return []

        try:
            completeness = float(parsed.get('completeness', 0))
            feasibility = float(parsed.get('feasibility', 0))
        except (TypeError, ValueError):
            return []

        if completeness >= self.min_completeness_score and feasibility >= self.min_feasibility_score:
            return data
        return []
