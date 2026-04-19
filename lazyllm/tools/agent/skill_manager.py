import os
import re
import shlex
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from lazyllm import config, ModuleBase, thirdparty
from .file_tool import read_file as _read_file
from .shell_tool import shell_tool as _shell_tool

DEFAULT_SKILLS_DIR = os.path.join(config['home'], 'skills')
os.makedirs(DEFAULT_SKILLS_DIR, exist_ok=True)
config.add(
    'skills_dir', str, DEFAULT_SKILLS_DIR, 'SKILLS_DIR',
    description='The directory of skills, supports multiple directories separated by commas'
).add(
    'max_skill_md_bytes', int, 5 * 1024 * 1024, 'MAX_SKILL_MD_BYTES',
    description='The maximum size of SKILL.md that can be loaded by default'
)

# Filename convention for cloud FS skills (no extension).
# A node/file whose name starts with this prefix is treated as a skill definition.
CLOUD_SKILL_PREFIX = 'SKILL'

SKILLS_PROMPT = '''
## Skills Guide
You have access to a skills library that encodes specialized workflows and domain knowledge.

### How to Use Skills (Progressive Disclosure)
You can see the name/description above, but only fetch a skill's full instructions when it's relevant.

1) **Identify relevance**: If a skill's description matches the task, consider using it.
2) **Get the skill's full usage**: Call `get_skill` to retrieve the complete workflow.
3) **Follow the workflow strictly**: After you obtain the full usage,
execute the workflow steps, constraints, and examples in order.
4) **Adapt workflow to the current task**: Before execution, map the workflow
to the user's actual goal, constraints, and available inputs; do not apply
steps blindly.
5) **Load support files only when needed**: Use `read_reference` to read referenced files on demand.
6) **Run helper scripts only when required**: Use `run_script` with absolute paths and request approval if risky.

### Reference and Script
- **Reference**: Documentation and guidance files (e.g., design notes,
  domain rules, templates). Read them with `read_reference` to understand
  how to execute the workflow correctly.
- **Script**: Executable helpers provided by the skill. Prefer running these
  scripts with `run_script` instead of writing new programs from scratch.
- If a suitable script already exists in the skill, use it first. Only write
  new code when the existing scripts cannot satisfy the task.

### When Skills Help
- The user asks for a structured or repeatable process
- You need specialized domain context or best practices
- The skill provides a proven workflow for complex tasks

### Script Execution
Skills may include Python or shell scripts. Prefer `run_script` for scripts provided by the selected skill.
Use `shell_tool` only when needed, and always use absolute paths.

### Example
User: "Research the latest developments in quantum computing."

1) See a "web‑research" skill in the list
2) Call `get_skill` to fetch its full usage
3) Follow the research workflow (search → organize → synthesize)
4) Use `read_reference` and `run_script` if the workflow calls for them

Skills improve reliability and consistency. If a skill applies, use it.
'''


_META_REQUIRED_FIELDS = {
    'name',
    'description',
}


def _local_fs():
    return thirdparty.fsspec.implementations.local.LocalFileSystem()


class SkillManager(ModuleBase):
    def __init__(self, dir: Optional[str] = None, skills: Optional[Iterable[str]] = None,
                 max_skill_md_bytes: Optional[int] = None, fs=None):
        super().__init__(return_trace=False)
        self._fs = fs or _local_fs()
        self._is_local = fs is None
        self._skills_dir = self._parse_dirs(dir) if dir else (
            self._parse_dirs(config['skills_dir']) if self._is_local else []
        )
        self._validate_fs_dir_consistency(fs, self._skills_dir)
        self._skills_expected = self._parse_skills(skills)
        self._max_skill_md_bytes = max_skill_md_bytes or config['max_skill_md_bytes']
        self._skills_index: Dict[str, Dict] = {}
        self._skills_selected: List[str] = []

    @staticmethod
    def _extract_protocol(path: str) -> Optional[str]:
        m = re.match(r'^([a-zA-Z][a-zA-Z0-9+\-.]*)(@[^:/]+)?:/', path)
        return m.group(1).lower() if m else None

    @staticmethod
    def _validate_fs_dir_consistency(fs, dirs: List[str]) -> None:
        from lazyllm.tools.fs.client import _FSRouter
        if fs is None or isinstance(fs, _FSRouter):
            return
        fs_protocol = getattr(fs, '_fs_protocol_key', None)
        for path in dirs:
            path_protocol = SkillManager._extract_protocol(path)
            if fs_protocol:
                # Known FS: path without protocol is fine (treated as this FS's protocol);
                # path with a different protocol is an error.
                if path_protocol is not None and path_protocol != fs_protocol:
                    raise ValueError(
                        f'dir protocol {path_protocol!r} does not match fs protocol {fs_protocol!r}. '
                        f'Use \'{fs_protocol}:/your/path\' or a bare path, or pass the matching FS instance.'
                    )
            else:
                # Unknown third-party FS: cannot validate a protocol prefix, so reject it.
                if path_protocol is not None:
                    raise ValueError(
                        f'dir {path!r} has a protocol prefix {path_protocol!r}, but the provided fs '
                        f'{type(fs).__name__!r} has no _fs_protocol_key. '
                        f'Use a bare path (without protocol prefix) for this FS.'
                    )

    @staticmethod
    def _parse_dirs(dir_value: Optional[str]) -> List[str]:
        if not dir_value:
            return []
        dirs = [d.strip() for d in dir_value.split(',') if d.strip()] if isinstance(dir_value, str) else list(dir_value)
        seen = set()
        result = []
        for d in dirs:
            if not d:
                continue
            # Keep cloud paths (protocol:/ prefix) as-is; expand local paths
            path = d if re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*:/', d) else os.path.abspath(os.path.expanduser(d))
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result

    @staticmethod
    def _parse_skills(skills: Optional[Iterable[str]]) -> List[str]:
        if skills is None:
            return []
        items = [s.strip() for s in skills.split(',') if s.strip()] if isinstance(skills, str) else list(skills)
        seen: set = set()
        result = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _fs_read(self, path: str) -> str:
        with self._fs.open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _fs_getsize(self, path: str) -> int:
        return self._fs.info(path).get('size', 0)

    def _fs_listdir(self, path: str) -> List[Dict]:
        try:
            return self._fs.ls(path, detail=True)
        except Exception:
            return []

    def _fs_join(self, base: str, name: str) -> str:
        if self._is_local:
            return os.path.join(base, name)
        return base.rstrip('/') + '/' + name

    def _iter_skill_files(self) -> Iterable[Tuple[str, str]]:
        for base_dir in self._skills_dir:
            stack = [base_dir]
            while stack:
                cur = stack.pop()
                entries = self._fs_listdir(cur)
                skill_node = None
                subdirs = []
                for entry in entries:
                    name = entry.get('name', '')
                    basename = name.rsplit('/', 1)[-1]
                    etype = entry.get('type', 'file')
                    if etype not in ('directory', 'dir'):
                        # Local: match 'SKILL.md'; cloud: match any name starting with CLOUD_SKILL_PREFIX
                        if (self._is_local and basename == 'SKILL.md') or \
                                (not self._is_local and basename.startswith(CLOUD_SKILL_PREFIX)):
                            skill_node = name
                    elif etype in ('directory', 'dir'):
                        subdirs.append(name)
                if skill_node:
                    yield cur, skill_node
                else:
                    for subdir in reversed(subdirs):
                        stack.append(subdir)

    @staticmethod
    def _extract_yaml_meta(text: str) -> Optional[dict]:
        lines = text.splitlines()
        start_idx = end_idx = None
        for idx, line in enumerate(lines):
            if line.strip() == '---':
                if start_idx is None:
                    start_idx = idx
                    continue
                end_idx = idx
                break
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return None
        try:
            meta = yaml.safe_load('\n'.join(lines[start_idx + 1:end_idx])) or {}
        except yaml.YAMLError:
            return None
        return meta if isinstance(meta, dict) else None

    @staticmethod
    def _is_meta_valid(meta: dict) -> bool:
        return meta is not None and _META_REQUIRED_FIELDS.issubset(meta.keys())

    def _load_skills_index(self) -> None:
        if self._skills_index:
            return
        seen: set = set()
        for skill_dir, skill_md in self._iter_skill_files():
            if self._fs_getsize(skill_md) > self._max_skill_md_bytes:
                continue
            try:
                content = self._fs_read(skill_md)
            except Exception:
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
            self._skills_selected = [n for n in self._skills_expected if n in self._skills_index]
        else:
            self._skills_selected = [
                n for n, info in self._skills_index.items() if not info.get('disable-model-invocation')
            ]

    def list_skill(self) -> str:
        self._load_skills_index()
        lines = ['# Skills', '', '## Skill Locations']
        lines.extend([f'- {path}' for path in self._skills_dir] or ['- (none)'])
        lines += ['', '## Available Skills']
        if not self._skills_index:
            lines.append('- (none)')
            return '\n'.join(lines)
        for name, info in self._skills_index.items():
            desc = (info.get('description', '') or '')[:1024]
            lines += [f'- **{name}**', f'  - {desc}', f'  - Path: {info.get("path")}']
        return '\n'.join(lines)

    def build_prompt(self, task: str) -> str:
        self._load_skills_index()
        selected = self._selector(task) or list(self._skills_index.keys())
        skills_list = self._format_skills_list(selected)
        lines = ['**Skills Directory**']
        if self._skills_dir:
            lines.append(self._format_skills_locations())
        lines += ['**Available Skills**', skills_list or '- (none)']
        return '\n'.join(lines)

    def _available_skills_text(self, task: str) -> str:
        return self.build_prompt(task)

    def wrap_input(self, input, task: str):
        available = self._available_skills_text(task)
        if not available:
            return input
        if isinstance(input, dict):
            if 'available_skills' in input:
                return input
            ret = dict(input)
            if 'input' not in ret:
                ret['input'] = ret.pop('content', task)
            else:
                ret.pop('content', None)
            ret.pop('role', None)
            ret['available_skills'] = available
            return ret
        if isinstance(input, str):
            return {'input': input, 'available_skills': available}
        return input

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'y', 'on')
        return bool(value) if value is not None else False

    def _selector(self, task: str) -> List[str]:
        self._load_skills_index()
        if self._skills_expected:
            return list(self._skills_selected)
        candidates = self._skills_selected
        # TODO: Use BM25-based ranking when rag dependencies are available.
        return list(candidates) if task and candidates else list(candidates)

    def get_skill(self, name: str, allow_large: bool = False) -> Dict[str, str]:
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        skill_md = info['skill_md']
        size = self._fs_getsize(skill_md)
        if size > self._max_skill_md_bytes and not allow_large:
            return {'status': 'too_large', 'name': name, 'path': skill_md,
                    'size': size, 'limit': self._max_skill_md_bytes}
        try:
            content = self._fs_read(skill_md)
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}
        return {'status': 'ok', 'name': name, 'path': skill_md, 'content': content}

    def read_file(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        base = info['path']
        path = self._fs_join(base, rel_path)
        if not self._is_local:
            try:
                return {'status': 'ok', 'path': path, 'content': self._fs_read(path)}
            except Exception as e:
                return {'status': 'error', 'path': path, 'error': str(e)}
        return _read_file(path, root=base, **kwargs)

    def run_script(self, name: str, rel_path: str, args: Optional[List[str]] = None,
                   allow_unsafe: bool = False, cwd: Optional[str] = None) -> Dict[str, str]:
        self._load_skills_index()
        info = self._skills_index.get(name)
        if not info:
            return {'status': 'missing', 'name': name}
        if not self._is_local:
            return {'status': 'error', 'error': 'run_script is not supported for cloud FS skills'}
        base = info['path']
        script_path = os.path.join(base, rel_path)
        if not os.path.exists(script_path):
            return {'status': 'missing', 'path': script_path}
        ext = os.path.splitext(script_path)[1].lower()
        cmd = ['python' if ext == '.py' else 'bash' if ext in ('.sh', '.bash') else 'sh', script_path]
        if args:
            cmd.extend(args)
        return _shell_tool(' '.join(shlex.quote(p) for p in cmd), cwd=cwd or base, allow_unsafe=allow_unsafe)

    def read_reference(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        return self.read_file(name=name, rel_path=rel_path, **kwargs)

    def get_skill_tools(self) -> List:
        return [self._build_get_skill_tool(), self._build_read_reference_tool(), self._build_run_script_tool()]

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
            if info:
                desc = (info.get('description', '') or '')[:1024]
                lines.append(f'- {name}: {desc} (path: {info.get("path")})')
        return '\n'.join(lines)

    def _format_skills_locations(self) -> str:
        return '\n'.join(f'- {path}' for path in self._skills_dir)
