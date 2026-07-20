import os
import posixpath
import re
import shlex
import subprocess
import tempfile
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from lazyllm import config, LOG, ModuleBase
from lazyllm.thirdparty import fsspec
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
SKILL_PREFIX = 'SKILL'

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

### Skill Selection Restraint
Do not load a skill merely because it could improve or add structure to the answer.
For simple questions, normal how-to guidance, and ordinary recommendations, answer directly
or use the relevant tool without loading a heavyweight workflow. A skill whose own description
says not to trigger on simple requests must not be loaded for those requests.

### Reference and Script Tools (Strict Constraint)
**CRITICAL — Read Before Using `read_reference` or `run_script`:**

These two tools have a hard prerequisite and a hard path rule:

**Prerequisite**: You MUST call `get_skill` for the skill first and receive its
SKILL.md content before you can use `read_reference` or `run_script`.

**Path Rule**: The `rel_path` argument MUST be copied verbatim from the SKILL.md
body. You are FORBIDDEN from fabricating, guessing, or extrapolating paths.
Examples of FORBIDDEN path fabrication:
  - Making up paths like `scripts/create_spring_prose.sh` that do not appear in SKILL.md
  - Assuming a file exists because "most skills have it" (e.g., `scripts/setup.sh`)
  - Constructing paths from the skill name (e.g., `scripts/<skill_name>.py`)
  - Trying common filenames (e.g., `README.md`, `docs/guide.md`) unless explicitly listed

If the SKILL.md does not contain any explicit reference or script path, these
tools MUST NOT be called for that skill — use your other available tools instead.

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
Skills may include Python or shell scripts. Prefer `run_script` for scripts explicitly listed by the selected skill.

### Example
User: "Research the latest developments in quantum computing."

1) See a "web‑research" skill in the list
2) Call `get_skill` to fetch its full usage
3) Follow the research workflow (search → organize → synthesize)
4) Use `read_reference` and `run_script` only if the SKILL.md explicitly names them

Skills improve reliability and consistency. If a skill applies, use it.
'''


_META_REQUIRED_FIELDS = {
    'name',
    'description',
}


class SkillManager(ModuleBase):
    def __init__(self, dir: Optional[str] = None, skills: Optional[Iterable[str]] = None,
                 max_skill_md_bytes: Optional[int] = None, fs=None):
        super().__init__(return_trace=False)
        self._fs = fs or fsspec.implementations.local.LocalFileSystem()
        self._skills_dir = self._parse_dirs(dir or config['skills_dir'], fs=fs)
        self._validate_fs_dir_consistency(fs, self._skills_dir)
        self._skills_expected = self._parse_skills(skills)
        self._max_skill_md_bytes = max_skill_md_bytes or config['max_skill_md_bytes']
        self._skills_index: Dict[str, Dict] = {}
        self._skills_selected: List[str] = []
        self._skills_index_lock = threading.Lock()

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
    def _is_fs_router(fs) -> bool:
        from lazyllm.tools.fs.client import _FSRouter
        return isinstance(fs, _FSRouter)

    @staticmethod
    def _is_local_fs(fs) -> bool:
        if fs is None or SkillManager._is_fs_router(fs):
            return True
        if isinstance(fs, fsspec.implementations.local.LocalFileSystem):
            return True
        protocol = getattr(fs, 'protocol', None)
        protocol_key = getattr(fs, '_fs_protocol_key', None)
        protocols = protocol if isinstance(protocol, (tuple, list, set)) else [protocol]
        return protocol_key == 'file' or 'file' in protocols

    @staticmethod
    def _parse_dirs(dir_value: Optional[str], fs=None) -> List[str]:
        if not dir_value:
            return []
        dirs = [d.strip() for d in dir_value.split(',') if d.strip()] if isinstance(dir_value, str) else list(dir_value)
        seen = set()
        result = []
        expand_bare_paths = SkillManager._is_local_fs(fs)
        for d in dirs:
            if not d:
                continue
            # Keep cloud paths (protocol:/ prefix) as-is; expand local paths.
            # Use the same regex as _extract_protocol for consistency.
            is_cloud_path = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*(@[^:/]+)?:/', d))
            path = d if is_cloud_path or not expand_bare_paths else os.path.abspath(os.path.expanduser(d))
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

    def _fs_getsize(self, path: str) -> Optional[int]:
        try:
            size = self._fs.info(path).get('size')
        except Exception:
            return None
        try:
            return int(size) if size is not None else None
        except (TypeError, ValueError):
            return None

    def _content_exceeds_limit(self, content: str) -> bool:
        return len(content.encode('utf-8')) > self._max_skill_md_bytes

    def _fs_listdir(self, path: str) -> List[Dict]:
        try:
            return self._fs.ls(path, detail=True)
        except Exception:
            return []

    def _fs_join(self, base: str, name: str) -> str:
        return base.rstrip('/') + '/' + name

    @staticmethod
    def _normalize_skill_rel_path(path: str, label: str = 'rel_path') -> str:
        raw = str(path or '').strip()
        if not raw:
            raise ValueError(f'{label} must be a non-empty relative path inside the skill directory.')
        if raw.startswith('/') or os.path.isabs(raw) or '\\' in raw or re.match(r'^[A-Za-z]:[/\\]', raw):
            raise ValueError(f'{label} must be a relative POSIX path inside the skill directory.')
        parts = raw.split('/')
        if any(part in ('', '.', '..') for part in parts):
            raise ValueError(f"{label} must not contain empty, '.', or '..' path segments.")
        normalized = posixpath.normpath(raw)
        if normalized in ('', '.') or normalized == '..' or normalized.startswith('../'):
            raise ValueError(f'{label} must stay inside the skill directory.')
        return normalized

    @staticmethod
    def _resolve_local_skill_child(base: str, rel_path: str, label: str = 'rel_path') -> str:
        base_real = os.path.realpath(os.path.abspath(base))
        target = os.path.realpath(os.path.abspath(os.path.join(base_real, *rel_path.split('/'))))
        if os.path.commonpath([base_real, target]) != base_real:
            raise ValueError(f'{label} must stay inside the skill directory.')
        return target

    @classmethod
    def _resolve_run_cwd(cls, base: str, cwd: Optional[str]) -> str:
        base_real = os.path.realpath(os.path.abspath(base))
        if cwd is None or str(cwd).strip() in ('', '.'):
            return base_real
        raw = str(cwd).strip()
        if os.path.isabs(raw):
            target = os.path.realpath(os.path.abspath(raw))
        else:
            rel_cwd = cls._normalize_skill_rel_path(raw, label='cwd')
            target = cls._resolve_local_skill_child(base_real, rel_cwd, label='cwd')
        if os.path.commonpath([base_real, target]) != base_real:
            raise ValueError('cwd must stay inside the skill directory.')
        return target

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
                    full_path = name if '/' in name else self._fs_join(cur, basename)
                    etype = entry.get('type', 'file')
                    if etype not in ('directory', 'dir'):
                        # match any name starting with SKILL_PREFIX
                        if basename.startswith(SKILL_PREFIX):
                            skill_node = full_path
                    elif etype in ('directory', 'dir'):
                        subdirs.append(full_path)
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
    def _validate_meta(meta: Optional[dict]) -> Optional[Dict[str, str]]:
        if not isinstance(meta, dict):
            return {
                'code': 'frontmatter_not_mapping',
                'expected_type': 'dict',
                'actual_type': type(meta).__name__,
            }
        for field in sorted(_META_REQUIRED_FIELDS):
            if field not in meta:
                return {
                    'code': 'metadata_field_missing',
                    'field': field,
                }
            value = meta[field]
            if not isinstance(value, str):
                return {
                    'code': 'metadata_field_type_error',
                    'field': field,
                    'expected_type': 'str',
                    'actual_type': type(value).__name__,
                }
            if not value.strip():
                return {
                    'code': 'metadata_field_value_error',
                    'field': field,
                }
        return None

    def _skill_key_from_dir(self, skill_dir: str) -> str:
        normalized_dir = self._normalize_index_path(skill_dir)
        for base_dir in sorted(self._skills_dir, key=len, reverse=True):
            normalized_base = self._normalize_index_path(base_dir)
            if normalized_dir == normalized_base:
                return normalized_dir.rsplit('/', 1)[-1]
            prefix = normalized_base.rstrip('/') + '/'
            if normalized_dir.startswith(prefix):
                return normalized_dir[len(prefix):].strip('/')
        return normalized_dir.rsplit('/', 1)[-1]

    @classmethod
    def _normalize_index_path(cls, path: str) -> str:
        raw = str(path or '').replace('\\', '/').rstrip('/')
        if cls._extract_protocol(raw):
            return raw
        return os.path.abspath(os.path.expanduser(raw)).replace('\\', '/').rstrip('/')

    def _load_skills_index(self) -> None:
        if self._skills_index:
            return
        with self._skills_index_lock:
            if self._skills_index:
                return
            skills_index: Dict[str, Dict] = {}
            for skill_dir, skill_md in self._iter_skill_files():
                try:
                    size = self._fs_getsize(skill_md)
                    if size is not None and size > self._max_skill_md_bytes:
                        continue
                    content = self._fs_read(skill_md)
                    if size is None and self._content_exceeds_limit(content):
                        continue
                    meta = self._extract_yaml_meta(content)
                    validation_error = self._validate_meta(meta)
                    if validation_error:
                        details = ' '.join(
                            f'{detail_key}={detail_value}'
                            for detail_key, detail_value in validation_error.items()
                        )
                        LOG.warning(
                            f'event=skill_load_skipped {details} skill_md={skill_md!r}'
                        )
                        continue
                    name = meta['name']
                    key = self._skill_key_from_dir(skill_dir)
                    if not key or key in skills_index:
                        continue
                    skills_index[key] = {
                        'key': key,
                        'name': name,
                        'description': meta['description'],
                        'argument-hint': meta.get('argument-hint', ''),
                        'disable-model-invocation': self._to_bool(meta.get('disable-model-invocation', False)),
                        'user-invocable': self._to_bool(meta.get('user-invocable', True)),
                        'allowed-tools': meta.get('allowed-tools'),
                        'source': self._extract_protocol(skill_dir) or 'file',
                        'path': skill_dir,
                        'skill_md': skill_md,
                        'raw_meta': meta,
                    }
                except Exception as exc:
                    LOG.warning(
                        'event=skill_load_skipped code=unexpected_skill_load_error '
                        f'error_type={type(exc).__name__} skill_md={skill_md!r}'
                    )
                    continue
            self._skills_index = skills_index
            if self._skills_expected:
                self._skills_selected = [
                    key for key in (self._resolve_skill_ref(ref, self._skills_index.keys())[0]
                                    for ref in self._skills_expected)
                    if key
                ]
            else:
                self._skills_selected = [
                    key for key, info in self._skills_index.items() if not info.get('disable-model-invocation')
                ]

    def _resolve_skill_ref(self, ref: str, keys: Iterable[str]) -> Tuple[Optional[str], Optional[Dict]]:
        name = str(ref or '').strip()
        if not name:
            return None, {'status': 'missing', 'name': name}
        key_list = list(keys)
        if name in key_list:
            return name, None
        matches = [
            key for key in key_list
            if self._skills_index[key].get('name') == name
            or key.rsplit('/', 1)[-1] == name
        ]
        if not matches:
            return None, {'status': 'missing', 'name': name}
        if len(matches) > 1:
            return None, {
                'status': 'ambiguous',
                'name': name,
                'matches': sorted(matches),
                'error': (
                    f'Ambiguous skill name {name!r}; use the full skill key '
                    f'such as {sorted(matches)[0]!r}.'
                ),
            }
        return matches[0], None

    def _visible_skill_keys(self) -> List[str]:
        self._load_skills_index()
        if self._skills_selected:
            return [key for key in self._skills_selected if key in self._skills_index]
        if self._skills_expected:
            return []
        return [
            key for key, info in self._skills_index.items()
            if not info.get('disable-model-invocation')
        ]

    def _visible_skills_index(self) -> Dict[str, Dict]:
        return {
            key: self._skills_index[key]
            for key in self._visible_skill_keys()
            if key in self._skills_index
        }

    def _get_visible_skill_info(self, name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        visible_keys = self._visible_skill_keys()
        key, error = self._resolve_skill_ref(name, visible_keys)
        if error:
            return None, error
        return self._skills_index.get(key), None

    def list_skill(self) -> str:
        visible_skills = self._visible_skills_index()
        lines = ['# Skills', '', '## Skill Locations']
        lines.extend([f'- {path}' for path in self._skills_dir] or ['- (none)'])
        lines += ['', '## Available Skills']
        if not visible_skills:
            lines.append('- (none)')
            return '\n'.join(lines)
        for name, info in visible_skills.items():
            desc = (info.get('description', '') or '')[:1024]
            lines += [
                f'- **{name}**',
                f'  - Name: {info.get("name")}',
                f'  - {desc}',
                f'  - Source: {info.get("source", "file")}',
                f'  - Path: {info.get("path")}',
            ]
        return '\n'.join(lines)

    def build_prompt(self) -> str:
        skills_list = self._format_skills_list(self._visible_skill_keys())
        lines = ['**Skills Directory**']
        if self._skills_dir:
            lines.append(self._format_skills_locations())
        lines += ['**Available Skills**', skills_list or '- (none)']
        return f'{SKILLS_PROMPT}\n\n' + '\n'.join(lines)

    def describe_prompt(self) -> List[Dict[str, str]]:
        '''Return model-facing skill prompt parts for context observability.'''
        visible_keys = self._visible_skill_keys()
        directory_lines = ['**Skills Directory**']
        if self._skills_dir:
            directory_lines.append(self._format_skills_locations())
        directory_lines.append('**Available Skills**')
        parts = [{
            'item_id': 'skills_usage_rules',
            'title': 'Skill usage rules',
            'source': 'skill.runtime',
            'content': f'{SKILLS_PROMPT}\n\n' + '\n'.join(directory_lines) + ('\n' if visible_keys else ''),
            'content_kind': 'instruction',
        }]
        for index, key in enumerate(visible_keys):
            info = self._skills_index.get(key)
            if not info:
                continue
            description = (info.get('description', '') or '')[:1024]
            content = (
                f'- {key}: {description} (source: {info.get("source", "file")}, '
                f'path: {info.get("path")})'
            )
            if index < len(visible_keys) - 1:
                content += '\n'
            parts.append({
                'item_id': f'skill_{key}',
                'title': str(info.get('name') or key),
                'source': str(info.get('source') or 'skill.registry'),
                'content': content,
                'content_kind': 'reference',
            })
        return parts

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'y', 'on')
        return bool(value) if value is not None else False

    def get_skill(self, name: str, allow_large: bool = False) -> Dict[str, str]:
        info, error = self._get_visible_skill_info(name)
        if error:
            return error
        if not info:
            return {'status': 'missing', 'name': name}
        skill_md = info['skill_md']
        size = self._fs_getsize(skill_md)
        if size is not None and size > self._max_skill_md_bytes and not allow_large:
            return {'status': 'too_large', 'name': name, 'path': skill_md,
                    'size': size, 'limit': self._max_skill_md_bytes}
        try:
            content = self._fs_read(skill_md)
        except Exception as e:
            return {'status': 'error', 'name': name, 'error': str(e)}
        if size is None:
            size = len(content.encode('utf-8'))
        if size > self._max_skill_md_bytes and not allow_large:
            return {'status': 'too_large', 'name': name, 'path': skill_md,
                    'size': size, 'limit': self._max_skill_md_bytes}
        return {'status': 'ok', 'name': name, 'path': skill_md, 'content': content}

    def read_file(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        info, error = self._get_visible_skill_info(name)
        if error:
            return error
        if not info:
            return {'status': 'missing', 'name': name}
        try:
            normalized_rel_path = self._normalize_skill_rel_path(rel_path)
        except ValueError as exc:
            return {'status': 'error', 'name': name, 'error': str(exc)}
        base = info['path']
        path = self._fs_join(base, normalized_rel_path)
        try:
            return {'status': 'ok', 'path': path, 'content': self._fs_read(path)}
        except Exception as e:
            return {'status': 'error', 'path': path, 'error': str(e)}

    def _materialize_script_base(self, base: str, rel_path: str) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
        remote_script_path = self._fs_join(base, rel_path)
        if self._is_local_fs(self._fs) and not self._extract_protocol(remote_script_path):
            return base, None
        temp_dir = tempfile.TemporaryDirectory(prefix='lazyllm-skill-')
        try:
            materialized = self._fs.materialize_dir(base, temp_dir.name) or {}
        except Exception as exc:
            temp_dir.cleanup()
            raise RuntimeError(f'Failed to materialize remote skill directory {base!r}: {exc}') from exc
        return str(materialized.get('local_dir') or temp_dir.name), temp_dir

    @staticmethod
    def _build_script_command(script_path: str, args: Optional[List[str]]) -> List[str]:
        ext = os.path.splitext(script_path)[1].lower()
        runner = 'python' if ext == '.py' else 'bash' if ext in ('.sh', '.bash') else 'sh'
        cmd = [runner, script_path]
        if args:
            cmd.extend(args)
        return cmd

    def _run_script_exception(self, name: str, rel_path: str, cwd: Optional[str],
                              run_cwd: Optional[str], exc: Exception) -> Dict[str, str]:
        error_cwd = run_cwd or cwd
        extra = {}
        if isinstance(exc, ValueError):
            error_cwd = cwd
            message = f'Invalid run_script path argument: {exc}'
        elif isinstance(exc, FileNotFoundError):
            message = f'run_script filesystem path not found: {exc}'
        elif isinstance(exc, subprocess.TimeoutExpired):
            message = f'run_script timed out after {exc.timeout} seconds.'
            extra['timeout'] = exc.timeout
        else:
            message = f'run_script execution failed: {exc}'
        return {
            'status': 'error',
            'name': name,
            'rel_path': rel_path,
            'cwd': error_cwd,
            'error_type': exc.__class__.__name__,
            'error': message,
            **extra,
        }

    def run_script(self, name: str, rel_path: str, args: Optional[List[str]] = None,
                   allow_unsafe: bool = False, cwd: Optional[str] = None) -> Dict[str, str]:
        info, error = self._get_visible_skill_info(name)
        if error:
            return error
        if not info:
            return {'status': 'missing', 'name': name}
        try:
            normalized_rel_path = self._normalize_skill_rel_path(rel_path)
        except ValueError as exc:
            return {'status': 'error', 'name': name, 'error': str(exc)}
        if not normalized_rel_path.startswith('scripts/'):
            return {
                'status': 'error',
                'name': name,
                'rel_path': normalized_rel_path,
                'error_type': 'InvalidRelPath',
                'error': 'run_script rel_path must be under scripts/.',
            }
        base = info['path']
        temp_dir = None
        run_cwd = None
        try:
            base, temp_dir = self._materialize_script_base(base, normalized_rel_path)
            script_path = self._resolve_local_skill_child(base, normalized_rel_path)
            run_cwd = self._resolve_run_cwd(base, cwd)
            script_exists = os.path.exists(script_path) if temp_dir is not None else self._fs.exists(script_path)
            if not script_exists:
                return {'status': 'missing', 'name': name, 'path': script_path, 'rel_path': normalized_rel_path}
            cmd = self._build_script_command(script_path, args)
            result = _shell_tool(' '.join(shlex.quote(p) for p in cmd), cwd=run_cwd, allow_unsafe=allow_unsafe)
            if result.get('status') == 'ok' and result.get('exit_code', 0) != 0:
                result['status'] = 'failed'
            return result
        except Exception as exc:
            return self._run_script_exception(name, normalized_rel_path, cwd, run_cwd, exc)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

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
                lines.append(
                    f'- {name}: {desc} (source: {info.get("source", "file")}, '
                    f'path: {info.get("path")})'
                )
        return '\n'.join(lines)

    def _format_skills_locations(self) -> str:
        return '\n'.join(f'- {path}' for path in self._skills_dir)
