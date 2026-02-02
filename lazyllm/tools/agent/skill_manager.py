import os
import shlex
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from lazyllm import config, ModuleBase
from .file_tool import read_file as _read_file
from .shell_tool import shell_tool as _shell_tool
from lazyllm.tools.rag.component.bm25 import BM25
from lazyllm.tools.rag.doc_node import DocNode

DEFAULT_SKILLS_DIR = os.path.join(config['home'], 'skills')
os.makedirs(DEFAULT_SKILLS_DIR, exist_ok=True)
config.add(
    'skills_dir', str, DEFAULT_SKILLS_DIR, 'SKILLS_DIR',
    description='The directory of skills, supports multiple directories separated by commas'
).add(
    'default_skills_topk', int, 10, 'DEFAULT_SKILLS_TOPK',
    description='The topk of skills to use when no skills are specified'
).add(
    'max_skill_md_bytes', int, 5 * 1024 * 1024, 'MAX_SKILL_MD_BYTES',
    description='The maximum size of SKILL.md that can be loaded by default'
)

SKILLS_PROMPT = '''
## Skills Guide
You have access to a skills library that encodes specialized workflows and domain knowledge.

{skills_locations}

**Available Skills**
{skills_list}

### How to Use Skills (Progressive Disclosure)
You can see the name/description above, but only fetch a skill’s full instructions when it’s relevant.

1) **Identify relevance**: If a skill’s description matches the task, consider using it.
2) **Get the skill’s full usage**: Call `get_skill` to retrieve the complete workflow.
3) **Follow the workflow**: After you obtain the full usage, execute the steps, constraints, and examples in order.
4) **Load support files only when needed**: Use `read_reference` to read referenced files on demand.
5) **Run helper scripts only when required**: Use `run_script` with absolute paths and request approval if risky.

### When Skills Help
- The user asks for a structured or repeatable process
- You need specialized domain context or best practices
- The skill provides a proven workflow for complex tasks

### Script Execution
Skills may include Python or shell scripts. Run them via `run_script` or `shell_tool`, and always use absolute paths.

### Example
User: “Research the latest developments in quantum computing.”

1) See a “web‑research” skill in the list
2) Call `get_skill` to fetch its full usage
3) Follow the research workflow (search → organize → synthesize)
4) Use `read_reference` and `run_script` if the workflow calls for them

Skills improve reliability and consistency. If a skill applies, use it.
'''


_META_REQUIRED_FIELDS = {
    'name',
    'description',
}


class SkillManager(ModuleBase):
    def __init__(self, dir: Optional[str] = None, skills: Optional[Iterable[str]] = None,
                 max_skill_md_bytes: Optional[int] = None):
        super().__init__(return_trace=False)
        self._skills_dir = self._parse_dirs(dir) if dir else self._parse_dirs(config['skills_dir'])
        self._skills_expected = self._parse_skills(skills)
        self._max_skill_md_bytes = max_skill_md_bytes or config['max_skill_md_bytes']
        self._skills_index: Dict[str, Dict] = {}
        self._skills_selected: List[str] = []

    @staticmethod
    def _parse_dirs(dir_value: Optional[str]) -> List[str]:
        if not dir_value:
            return []
        if isinstance(dir_value, str):
            dirs = [d.strip() for d in dir_value.split(',') if d.strip()]
        else:
            dirs = list(dir_value)
        result = []
        for d in dirs:
            if not d:
                continue
            path = os.path.abspath(os.path.expanduser(d))
            if path not in result:
                result.append(path)
        return result

    @staticmethod
    def _parse_skills(skills: Optional[Iterable[str]]) -> List[str]:
        if skills is None:
            return []
        if isinstance(skills, str):
            items = [s.strip() for s in skills.split(',') if s.strip()]
        else:
            items = [s for s in skills if s]
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _iter_skill_files(self) -> Iterable[Tuple[str, str]]:
        for base_dir in self._skills_dir:
            if not os.path.isdir(base_dir):
                continue
            stack = [base_dir]
            while stack:
                cur = stack.pop()
                try:
                    entries = sorted(os.listdir(cur))
                except OSError:
                    continue
                skill_md = os.path.join(cur, 'SKILL.md')
                if 'SKILL.md' in entries and os.path.isfile(skill_md):
                    yield cur, skill_md
                else:
                    subdirs = [os.path.join(cur, e) for e in entries if os.path.isdir(os.path.join(cur, e))]
                    for subdir in reversed(subdirs):
                        stack.append(subdir)

    @staticmethod
    def _extract_yaml_meta(text: str) -> Optional[dict]:
        lines = text.splitlines()
        start_idx = None
        end_idx = None
        for idx, line in enumerate(lines):
            if line.strip() == '---':
                if start_idx is None:
                    start_idx = idx
                    continue
                end_idx = idx
                break
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return None
        yaml_text = '\n'.join(lines[start_idx + 1:end_idx])
        try:
            meta = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError:
            return None
        return meta if isinstance(meta, dict) else None

    @staticmethod
    def _is_meta_valid(meta: dict) -> bool:
        return meta is not None and _META_REQUIRED_FIELDS.issubset(meta.keys())

    def _load_skills_index(self) -> None:
        if self._skills_index:
            return
        seen = set()
        for skill_dir, skill_md in self._iter_skill_files():
            try:
                size = os.path.getsize(skill_md)
            except OSError:
                continue
            if size > self._max_skill_md_bytes:
                continue
            try:
                with open(skill_md, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except OSError:
                continue
            meta = self._extract_yaml_meta(content)
            if not self._is_meta_valid(meta):
                continue
            name = meta.get('name')
            if not name or name in seen:
                continue
            seen.add(name)
            self._skills_index[name] = {
                'name': name,
                'description': meta.get('description', ''),
                'argument-hint': meta.get('argument-hint', ''),
                'disable-model-invocation': self._to_bool(meta.get('disable-model-invocation', False)),
                'user-invocable': self._to_bool(meta.get('user-invocable', True)),
                'allowed-tools': meta.get('allowed-tools'),
                'path': skill_dir,
                'skill_md': skill_md,
                'raw_meta': meta,
            }
        if self._skills_expected:
            self._skills_selected = [name for name in self._skills_expected if name in self._skills_index]
        else:
            self._skills_selected = [
                name for name, info in self._skills_index.items()
                if not info.get('disable-model-invocation')
            ]

    def list_skill(self) -> str:
        self._load_skills_index()
        lines = ['# Skills']
        if self._skills_dir:
            lines.append('')
            lines.append('## Skill Locations')
            lines.extend([f'- {path}' for path in self._skills_dir])
        else:
            lines.append('')
            lines.append('## Skill Locations')
            lines.append('- (none)')

        if not self._skills_index:
            lines.append('')
            lines.append('## Available Skills')
            lines.append('- (none)')
            return '\n'.join(lines)

        lines.append('')
        lines.append('## Available Skills')
        for name, info in self._skills_index.items():
            desc = info.get('description', '') or ''
            if len(desc) > 1024:
                desc = desc[:1024] + '...'
            path = info.get('path') or ''
            lines.append(f'- **{name}**')
            if desc:
                lines.append(f'  - {desc}')
            lines.append(f'  - Path: {path}')
        return '\n'.join(lines)

    def build_prompt(self, task: str) -> str:
        self._load_skills_index()
        selected = self._selector(task)
        if not selected:
            selected = list(self._skills_index.keys())
        skills_list = self._format_skills_list(selected)
        skills_locations = self._format_skills_locations()
        return SKILLS_PROMPT.format(skills_locations=skills_locations, skills_list=skills_list)

    def get_selected_skills(self) -> List[str]:
        self._load_skills_index()
        return list(self._skills_selected)

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'y', 'on')
        if value is None:
            return False
        return bool(value)

    def _selector(self, task: str) -> List[str]:
        self._load_skills_index()
        if self._skills_expected:
            return list(self._skills_selected)
        candidates = self._skills_selected
        if not task or not candidates:
            return candidates
        nodes: List[DocNode] = []
        for name in candidates:
            info = self._skills_index.get(name, {})
            text = f'{name}\n{info.get("description", "")}\n{info.get("argument-hint", "")}'
            node = DocNode(text=text, metadata={'skill_name': name})
            nodes.append(node)
        language = 'zh' if any('\u4e00' <= ch <= '\u9fff' for ch in task) else 'en'
        topk = min(config['default_skills_topk'], len(nodes))
        if topk <= 0:
            return []
        bm25 = BM25(nodes=nodes, language=language, topk=topk)
        results = bm25.retrieve(task, topk=topk)
        selected = []
        for node, _score in results:
            name = node.metadata.get('skill_name')
            if name and name not in selected:
                selected.append(name)
        return selected

    def get_skill(self, name: str, allow_large: bool = False) -> Dict[str, str]:
        '''Get full skill usage from SKILL.md.

        Args:
            name (str): Skill name.
            allow_large (bool, optional): Allow loading large SKILL.md. Defaults to False.
        '''
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        skill_md = info['skill_md']
        size = os.path.getsize(skill_md)
        if size > self._max_skill_md_bytes and not allow_large:
            return {
                'status': 'too_large',
                'name': name,
                'path': skill_md,
                'size': size,
                'limit': self._max_skill_md_bytes,
            }
        with open(skill_md, 'r', encoding='utf-8', errors='replace') as f:
            return {'status': 'ok', 'name': name, 'path': skill_md, 'content': f.read()}

    def read_file(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        '''Read a file inside a skill directory.

        Args:
            name (str): Skill name.
            rel_path (str): Relative file path inside the skill directory.
        '''
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        base = info['path']
        path = os.path.join(base, rel_path)
        return _read_file(path, root=base, **kwargs)

    def run_script(self, name: str, rel_path: str, args: Optional[List[str]] = None,
                   allow_unsafe: bool = False, cwd: Optional[str] = None) -> Dict[str, str]:
        '''Run a script inside a skill directory.

        Args:
            name (str): Skill name.
            rel_path (str): Relative script path inside the skill directory.
            args (list[str], optional): Script arguments.
            allow_unsafe (bool, optional): Allow execution. Defaults to False.
            cwd (str, optional): Working directory.
        '''
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        base = info['path']
        script_path = os.path.join(base, rel_path)
        if not os.path.exists(script_path):
            return {'status': 'missing', 'path': script_path}
        ext = os.path.splitext(script_path)[1].lower()
        if ext == '.py':
            cmd = ['python', script_path]
        elif ext in ('.sh', '.bash'):
            cmd = ['bash', script_path]
        else:
            cmd = ['sh', script_path]
        if args:
            cmd.extend(args)
        cmd_str = ' '.join(shlex.quote(part) for part in cmd)
        return _shell_tool(cmd_str, cwd=cwd or base, allow_unsafe=allow_unsafe)

    def read_reference(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        '''Read a reference file within a skill directory.'''
        return self.read_file(name=name, rel_path=rel_path, **kwargs)

    def get_skill_tools(self) -> List:
        return [
            self._build_get_skill_tool(),
            self._build_read_reference_tool(),
            self._build_run_script_tool(),
        ]

    def _build_get_skill_tool(self):
        def get_skill(name: str, allow_large: bool = False) -> dict:
            '''Get the full usage for a skill (SKILL.md).

            Args:
                name (str): Skill name.
                allow_large (bool, optional): Allow loading large SKILL.md. Defaults to False.
            '''
            return self.get_skill(name=name, allow_large=allow_large)
        return get_skill

    def _build_read_reference_tool(self):
        def read_reference(name: str, rel_path: str, **kwargs) -> dict:
            '''Read a reference file within a skill directory.

            Args:
                name (str): Skill name.
                rel_path (str): Relative file path inside the skill directory.
            '''
            return self.read_reference(name=name, rel_path=rel_path, **kwargs)
        return read_reference

    def _build_run_script_tool(self):
        def run_script(name: str, rel_path: str, args: Optional[List[str]] = None,
                       allow_unsafe: bool = False, cwd: Optional[str] = None) -> dict:
            '''Run a script within a skill directory.

            Args:
                name (str): Skill name.
                rel_path (str): Relative script path inside the skill directory.
                args (list[str], optional): Script arguments.
                allow_unsafe (bool, optional): Allow execution. Defaults to False.
                cwd (str, optional): Working directory.
            '''
            return self.run_script(name=name, rel_path=rel_path, args=args,
                                   allow_unsafe=allow_unsafe, cwd=cwd)
        return run_script

    def _format_skills_list(self, names: List[str]) -> str:
        lines = []
        for name in names:
            info = self._skills_index.get(name)
            if not info:
                continue
            desc = info.get('description', '') or ''
            if len(desc) > 1024:
                desc = desc[:1024] + '...'
            lines.append(f'- {name}: {desc} (path: {info.get("path")})')
        return '\n'.join(lines)

    def _format_skills_locations(self) -> str:
        if not self._skills_dir:
            return ''
        return '\n'.join([f'- {path}' for path in self._skills_dir])
