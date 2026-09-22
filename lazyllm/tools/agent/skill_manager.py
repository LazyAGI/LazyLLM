import json
import math
import os
import posixpath
import re
import subprocess
import tempfile
import threading
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from lazyllm import config, LOG, ModuleBase
from lazyllm.thirdparty import fsspec
from .missing_env import collect_missing_env_hints, format_missing_env_message
from .toolError import ToolExecutionError
from .toolsManager import fc_register

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
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20
MAX_SEARCH_CALLS = 2
SKILL_TOOL_MODE_FULL = 'full'
SKILL_TOOL_MODE_DISCOVERY = 'discovery'
_ASCII_TERM = re.compile(r'[a-z0-9]+')
_CJK_RUN = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]+')
_QUERY_STOP = frozenset({
    'skill', 'skills', 'the', 'and', 'for', 'with', 'from', 'that', 'this',
    'find', 'search', 'use', 'please', 'help', 'related',
    '技能', '帮我', '请帮', '一下', '相关', '找个', '找一下',
})
_QUERY_EXPAND = {
    '文章': ('article', 'writing', 'copywriting'),
    '写文章': ('article', 'writing', 'copywriting'),
    '写作': ('writing', 'article', 'copywriting'),
    '文案': ('copywriting', 'writing'),
    '图片': ('image', 'design', 'poster', 'visual'),
    '图像': ('image', 'design', 'visual'),
    '生图': ('image', 'design'),
    '海报': ('poster', 'image', 'design'),
    '配图': ('image', 'design', 'visual'),
    '视觉': ('visual', 'design', 'image'),
    '论文': ('paper', 'academic', 'research'),
    '文献': ('paper', 'academic', 'research'),
    'arxiv': ('paper', 'academic', 'research'),
    'anki': ('vocabulary', 'flashcard', 'review'),
}
_INSTRUCTIONAL_SPLIT = re.compile(
    r'(按照|步骤包括|包括：|包括:|Follow these|including:)',
    re.IGNORECASE,
)

SKILLS_PROMPT = '''
## Skills Guide
Skill usage:

1. Do not use skills for simple tasks that do not benefit from
   specialized instructions, workflows, or resources.

2. If the user explicitly names a skill, call `get_skill` directly.
   Never call `search_skill` before or after that lookup. If resolution
   fails, report the exact get_skill error code.

3. If the user asks to search, find, or recommend a skill first
   (先找 / 先搜索 / 推荐), call `search_skill` even when a catalog
   entry looks obvious. Catalog descriptions are routing metadata
   and must not substitute for SKILL.md.

4. Call `search_skill` once with a task query. It performs ranked
   retrieval and does not accept hard metadata filters.

5. After `search_skill`:
   - if executing the skill, call `get_skill`;
   - if the user only asked to find/recommend skills, report the
     candidates without loading them. Do not call `get_skill` just
     to verify a recommendation.

6. Do not repeatedly search to enumerate the library.
   A failed search is normally terminal unless the original query
   was clearly malformed.
   SkillManagementToolkit creates, installs, edits, or deletes
   skills. It is never a discovery substitute for `search_skill`.

7. Only access resources declared by a loaded skill.
'''


_META_REQUIRED_FIELDS = {
    'name',
    'description',
}


class SkillManager(ModuleBase):
    def __init__(self, dir: Optional[str] = None, skills: Optional[Iterable[str]] = None,
                 max_skill_md_bytes: Optional[int] = None, fs=None, sandbox=None,
                 prompt_skills: Optional[Iterable[str]] = None,
                 excluded_skills: Optional[Iterable[str]] = None,
                 skill_search: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 skill_tool_mode: str = SKILL_TOOL_MODE_FULL):
        super().__init__(return_trace=False)
        self._fs = fs or fsspec.implementations.local.LocalFileSystem()
        if sandbox is None:
            from lazyllm.tools.sandbox.sandbox_base import create_sandbox
            sandbox = create_sandbox()
        self._sandbox = sandbox
        self._skills_dir = self._parse_dirs(dir or config['skills_dir'], fs=fs)
        self._validate_fs_dir_consistency(fs, self._skills_dir)
        self._skills_expected = self._parse_skills(skills)
        # None keeps the default L1 catalog (all loadable skills). An empty list
        # hides individual L1 entries; get_skill / scripts still use ``skills``.
        self._prompt_skills = None if prompt_skills is None else self._parse_skills(prompt_skills)
        self._excluded_skills = set(self._parse_skills(excluded_skills))
        self._skill_search = skill_search
        mode = str(skill_tool_mode or SKILL_TOOL_MODE_FULL).strip()
        self._skill_tool_mode = (
            SKILL_TOOL_MODE_DISCOVERY if mode == SKILL_TOOL_MODE_DISCOVERY else SKILL_TOOL_MODE_FULL
        )
        self._max_skill_md_bytes = max_skill_md_bytes or config['max_skill_md_bytes']
        self._skills_index: Dict[str, Dict] = {}
        self._skills_selected: List[str] = []
        self._skills_index_lock = threading.Lock()
        self._search_count = 0
        self._searched_signatures: set = set()
        self._loaded_skills: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _extract_protocol(path: str) -> Optional[str]:
        # A Windows drive path (for example, ``C:/skills``) is local.  Without
        # this guard the generic protocol pattern mistakes the drive letter for
        # a filesystem protocol and sends local files through materialize_dir.
        if re.match(r'^[A-Za-z]:[/\\]', path):
            return None
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
            # Windows drive paths are local and are excluded by _extract_protocol.
            is_cloud_path = SkillManager._extract_protocol(d) is not None
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

    @staticmethod
    def _normalize_skill_ref(ref: str) -> str:
        name = unicodedata.normalize('NFKC', str(ref or '')).strip()
        return name.strip('`\'"“”‘’$')

    def _skill_ref_aliases(self, key: str) -> set:
        info = self._skills_index.get(key) or {}
        aliases = {
            key.casefold(),
            key.rsplit('/', 1)[-1].casefold(),
            str(info.get('name') or '').strip().casefold(),
        }
        return {item for item in aliases if item}

    def _ambiguous_skill_error(self, name: str, matches: List[str]) -> Dict[str, Any]:
        ordered = sorted(matches)
        return {
            'status': 'identifier_not_resolved',
            'code': 'identifier_not_resolved',
            'name': name,
            'matches': ordered,
            'error': (
                f'Ambiguous skill name {name!r}; use the full skill key '
                f'such as {ordered[0]!r}.'
            ),
        }

    def _resolve_skill_ref(self, ref: str, keys: Iterable[str]) -> Tuple[Optional[str], Optional[Dict]]:
        name = self._normalize_skill_ref(ref)
        if not name:
            return None, {
                'status': 'identifier_not_resolved',
                'code': 'identifier_not_resolved',
                'name': name,
                'error': 'Skill identifier is empty.',
            }
        key_list = [key for key in keys if not self._skills_index or key in self._skills_index]
        folded = name.casefold()
        exact = [key for key in key_list if key == name or key.casefold() == folded]
        if len(exact) == 1:
            return exact[0], None
        if len(exact) > 1:
            return None, self._ambiguous_skill_error(name, exact)
        if '/' in name:
            return None, {
                'status': 'skill_not_installed',
                'code': 'skill_not_installed',
                'name': name,
                'error': f'Skill {name!r} is not installed.',
            }
        matches = []
        for key in key_list:
            aliases = self._skill_ref_aliases(key)
            if folded in aliases:
                matches.append(key)
        if not matches:
            return None, {
                'status': 'skill_not_installed',
                'code': 'skill_not_installed',
                'name': name,
                'error': f'Skill {name!r} is not installed.',
            }
        if len(matches) > 1:
            return None, self._ambiguous_skill_error(name, matches)
        return matches[0], None

    def _resolve_loadable_skill(
        self, ref: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        self._load_skills_index()
        loadable = [
            key for key in self._visible_skill_keys()
            if key not in self._excluded_skills
            and not (self._skills_index.get(key) or {}).get('disable-model-invocation')
        ]
        key, loadable_error = self._resolve_skill_ref(ref, loadable)
        if key:
            return self._skills_index.get(key), None
        if loadable_error and loadable_error.get('code') == 'identifier_not_resolved':
            return None, loadable_error

        installed_key, installed_error = self._resolve_skill_ref(ref, self._skills_index.keys())
        if installed_error:
            return None, installed_error
        info = self._skills_index.get(installed_key)
        if installed_key in self._excluded_skills or info.get('disable-model-invocation'):
            return None, {
                'status': 'skill_exists_but_not_visible',
                'code': 'skill_exists_but_not_visible',
                'name': self._normalize_skill_ref(ref),
                'skill_key': installed_key,
                'error': f'Skill {installed_key!r} exists but is not visible in this context.',
            }
        if self._skills_expected and installed_key not in self._skills_selected:
            return None, {
                'status': 'skill_disabled',
                'code': 'skill_disabled',
                'name': self._normalize_skill_ref(ref),
                'skill_key': installed_key,
                'error': f'Skill {installed_key!r} is installed but disabled.',
            }
        return info, None

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

    def set_prompt_skills(self, skills: Optional[Iterable[str]] = None) -> None:
        '''Set the L1 prompt catalog without changing loadable scope.

        ``None`` restores the default catalog. An empty iterable injects no
        individual L1 entries; ``get_skill`` / ``read_reference`` / ``run_script``
        still resolve against the loadable ``skills`` list.
        '''
        self._prompt_skills = None if skills is None else self._parse_skills(skills)

    def _prompt_skill_keys(self) -> List[str]:
        visible = [key for key in self._visible_skill_keys() if key not in self._excluded_skills]
        if self._prompt_skills is None:
            return visible
        if not self._prompt_skills:
            return []
        resolved: List[str] = []
        seen = set()
        for ref in self._prompt_skills:
            key, _error = self._resolve_skill_ref(ref, visible)
            if key and key not in seen:
                seen.add(key)
                resolved.append(key)
        return resolved

    def _search_allowed_keys(self) -> set:
        return {key for key in self._visible_skill_keys() if key not in self._excluded_skills}

    @staticmethod
    def _search_limit(value) -> int:
        try:
            return max(1, min(MAX_SEARCH_LIMIT, int(value)))
        except (TypeError, ValueError):
            return DEFAULT_SEARCH_LIMIT

    def _search_signature(self, query: str) -> str:
        return json.dumps({'query': query}, sort_keys=True, ensure_ascii=True)

    def list_prompt_skills(self) -> Dict[str, Any]:
        skills = self._prompt_skill_keys()
        return {'status': 'ok', 'count': len(skills), 'skills': skills}

    def search_skill(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        *,
        enforce_budget: bool = True,
    ) -> Dict[str, Any]:
        cleaned = str(query or '').strip()
        if not cleaned:
            return {'status': 'error', 'error': 'query is required', 'hits': [], 'retry': False}
        signature = self._search_signature(cleaned)
        if enforce_budget and signature in self._searched_signatures:
            return {
                'status': 'error',
                'error': 'duplicate_search',
                'hits': [],
                'retry': False,
            }
        if enforce_budget and self._search_count >= MAX_SEARCH_CALLS:
            return {
                'status': 'error',
                'error': 'retrieval_budget_exhausted',
                'hits': [],
                'retry': False,
            }
        result = self._search_skill_catalog(
            {'query': cleaned},
            self._search_limit(limit),
        )
        if enforce_budget and result.get('status') != 'error':
            self._searched_signatures.add(signature)
            self._search_count += 1
        return result

    def _search_skill_catalog(
        self,
        request: Dict[str, Any],
        limit: int,
    ) -> Dict[str, Any]:
        allowed = self._search_allowed_keys()
        empty = {
            'status': 'ok',
            'query': request.get('query', ''),
            'hits': [],
            'scope': 'full_skill_library',
            'has_more': False,
        }
        if not allowed:
            return empty
        payload = dict(request)
        payload.update({
            'limit': min(limit + 1, MAX_SEARCH_LIMIT + 1),
            'exclude': sorted(self._excluded_skills),
            'allowed_skill_keys': sorted(allowed),
        })
        search = self._skill_search or self._local_skill_search
        try:
            result = search(payload) or {}
        except Exception as exc:
            return {'status': 'error', 'error': str(exc), 'hits': []}
        if result.get('status') == 'error':
            return {
                'status': 'error',
                'error': str(result.get('error') or 'skill search failed'),
                'hits': [],
            }
        raw_hits = result.get('skills') or result.get('hits')
        hits, seen = [], set()
        for item in raw_hits or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get('skill_key') or '').strip()
            if key not in allowed or key in seen:
                continue
            seen.add(key)
            description = self._routing_description_text(str(item.get('description') or ''), key)
            hits.append({
                'skill_key': key,
                'name': str(item.get('name') or key.rsplit('/', 1)[-1]),
                'description': description,
                'match_reason': str(item.get('match_reason') or description),
            })
        has_more = len(hits) > limit
        hits = hits[:limit]
        return {
            'status': 'ok',
            'scope': 'full_skill_library',
            'query': str(request.get('query') or ''),
            'hits': hits,
            'has_more': has_more,
        }

    def _local_skill_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._load_skills_index()
        query = str(payload.get('query') or '').strip()
        allowed = set(payload.get('allowed_skill_keys') or [])
        exclude = set(payload.get('exclude') or [])
        candidates = []
        for key in allowed:
            if key in exclude:
                continue
            info = self._skills_index.get(key) or {}
            candidates.append((key, info, self._search_fields(key, info)))
        q_norm = self._normalize_search_text(query)
        literals, expansions = self._query_terms(query)
        n = max(1, len(candidates))
        df: Dict[str, int] = {}
        for _key, _info, fields in candidates:
            blob = ' '.join(fields.values())
            seen = set()
            for term in (*literals, *expansions):
                if term in seen:
                    continue
                if self._term_in_text(term, blob):
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)
        scored = []
        for key, info, fields in candidates:
            score, reason, coverage = self._score_skill_query(
                key, info, fields, q_norm, literals, expansions, df, n,
            )
            if score <= 0:
                continue
            scored.append((score, coverage, {
                'skill_key': key,
                'name': info.get('name') or key.rsplit('/', 1)[-1],
                'description': self._routing_description(info),
                'match_reason': reason,
            }))
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]['skill_key']))
        try:
            requested = int(payload.get('limit'))
        except (TypeError, ValueError):
            requested = DEFAULT_SEARCH_LIMIT
        # Allow one extra hit so search_skill can set has_more at MAX_SEARCH_LIMIT.
        limit = max(1, min(MAX_SEARCH_LIMIT + 1, requested))
        return {'status': 'ok', 'hits': [item for _s, _c, item in scored[:limit]]}

    @staticmethod
    def _normalize_search_text(text: Any) -> str:
        return unicodedata.normalize('NFKC', str(text or '')).casefold()

    @classmethod
    def _join_meta_values(cls, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, list):
            return ' '.join(cls._normalize_search_text(item) for item in value if str(item).strip())
        return cls._normalize_search_text(value)

    @classmethod
    def _search_fields(cls, key: str, info: Dict[str, Any]) -> Dict[str, str]:
        meta = info.get('raw_meta') or {}
        field = str(meta.get('field') or (key.split('/', 1)[0] if '/' in key else ''))
        return {
            'key': cls._normalize_search_text(key.replace('/', ' ').replace('-', ' ').replace('_', ' ')),
            'name': cls._normalize_search_text(info.get('name') or ''),
            'aliases': ' '.join(filter(None, [
                cls._join_meta_values(meta.get('aliases')),
                cls._join_meta_values(meta.get('keywords')),
                cls._join_meta_values(meta.get('tags')),
            ])),
            'when': cls._normalize_search_text(meta.get('when_to_use') or ''),
            'description': cls._normalize_search_text(info.get('description') or ''),
            'field': cls._normalize_search_text(field),
        }

    @classmethod
    def _remember_term(cls, terms: List[str], term: str, *, normalize: bool = False) -> None:
        cleaned = cls._normalize_search_text(term) if normalize else term.strip()
        if cleaned and cleaned not in _QUERY_STOP and cleaned not in terms:
            terms.append(cleaned)

    @classmethod
    def _append_cjk_literals(cls, literals: List[str], run: str) -> None:
        if not run or run in _QUERY_STOP:
            return
        if len(run) < 2:
            cls._remember_term(literals, run)
            return
        for size in (2, 3):
            if len(run) < size:
                continue
            for index in range(len(run) - size + 1):
                cls._remember_term(literals, run[index:index + size])

    @classmethod
    def _literal_terms(cls, query: str) -> List[str]:
        literals: List[str] = []
        for token in _ASCII_TERM.findall(query):
            if len(token) >= 2:
                cls._remember_term(literals, token)
        for run in _CJK_RUN.findall(query):
            cls._append_cjk_literals(literals, run)
        return literals

    @classmethod
    def _expansion_terms(cls, query: str) -> List[str]:
        expansions: List[str] = []
        for source, targets in _QUERY_EXPAND.items():
            if source not in query:
                continue
            for target in targets:
                cls._remember_term(expansions, target, normalize=True)
        return expansions

    @classmethod
    def _query_terms(cls, query: str) -> Tuple[List[str], List[str]]:
        q = cls._normalize_search_text(query)
        return cls._literal_terms(q), cls._expansion_terms(q)

    @staticmethod
    def _term_in_text(term: str, text: str) -> bool:
        if not term or not text:
            return False
        if term.isascii():
            return re.search(rf'(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)', text) is not None
        return term in text

    @classmethod
    def _score_skill_query(
        cls,
        key: str,
        info: Dict[str, Any],
        fields: Dict[str, str],
        q_norm: str,
        literals: List[str],
        expansions: List[str],
        df: Dict[str, int],
        n: int,
    ) -> Tuple[float, str, float]:
        name = cls._normalize_search_text(info.get('name') or '')
        key_norm = cls._normalize_search_text(key)
        reason = fields.get('when') or fields.get('description') or name or key
        if q_norm and q_norm in {key_norm, name}:
            return 100.0, reason, 1.0
        weights = {
            'key': 12.0,
            'name': 12.0,
            'aliases': 9.0,
            'when': 6.0,
            'description': 4.0,
            'field': 3.0,
        }
        score = 0.0
        matched_literals = 0
        if q_norm and len(q_norm) >= 2 and (q_norm in key_norm or q_norm in name):
            score += 30.0
        for term in literals:
            idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
            field_weight = max(
                (weight for field, weight in weights.items() if cls._term_in_text(term, fields.get(field, ''))),
                default=0.0,
            )
            if field_weight:
                matched_literals += 1
                score += idf * field_weight
        for term in expansions:
            idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
            field_weight = max(
                (weight for field, weight in weights.items() if cls._term_in_text(term, fields.get(field, ''))),
                default=0.0,
            )
            if field_weight:
                score += idf * field_weight * 0.65
        coverage = (matched_literals / len(literals)) if literals else 0.0
        if literals:
            score += 6.0 * coverage
        has_signal = bool(literals or expansions) and score > 0
        if not has_signal:
            return 0.0, reason, 0.0
        return score, reason, coverage

    def _visible_skills_index(self) -> Dict[str, Dict]:
        return {
            key: self._skills_index[key]
            for key in self._visible_skill_keys()
            if key in self._skills_index
        }

    def _get_visible_skill_info(self, name: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        return self._resolve_loadable_skill(name)

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
        catalog_keys = self._prompt_skill_keys()
        skills_list = self._format_skills_list(
            catalog_keys, include_location=self._prompt_skills is None,
        )
        lines = ['**Skills Directory**']
        if self._skills_dir:
            lines.append(self._format_skills_locations())
        lines += ['**Available Skills**', skills_list or '- (none)']
        return f'{SKILLS_PROMPT}\n\n' + '\n'.join(lines)

    def describe_prompt(self) -> List[Dict[str, str]]:
        '''Return model-facing skill prompt parts for context observability.'''
        visible_keys = self._prompt_skill_keys()
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
            description = self._routing_description(info)
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
            return {
                'status': 'error',
                'code': str(error.get('code') or 'identifier_not_resolved'),
                'error': str(error.get('error') or 'Skill identifier could not be resolved.'),
                **({'skill_key': error['skill_key']} if error.get('skill_key') else {}),
                **({'matches': error['matches']} if error.get('matches') else {}),
                'retry': False,
            }
        if not info:
            return {
                'status': 'error',
                'code': 'skill_not_installed',
                'error': f'Skill {name!r} is not installed.',
                'retry': False,
            }
        cached = self._loaded_skills.get(info['key'])
        if cached:
            result = dict(cached)
            result['already_loaded'] = True
            return result
        skill_md = info['skill_md']
        size = self._fs_getsize(skill_md)
        if size is not None and size > self._max_skill_md_bytes and not allow_large:
            raise ToolExecutionError(
                f'Skill {name} is {size} bytes and exceeds the configured '
                f'{self._max_skill_md_bytes}-byte size limit. Set allow_large=True to read it.'
            )
        try:
            content = self._fs_read(skill_md)
        except Exception as e:
            raise ToolExecutionError(
                f'Failed to read skill {name} at {skill_md}: {e}'
            ) from e
        if size is None:
            size = len(content.encode('utf-8'))
        if size > self._max_skill_md_bytes and not allow_large:
            raise ToolExecutionError(
                f'Skill {name} is {size} bytes and exceeds the configured '
                f'{self._max_skill_md_bytes}-byte size limit. Set allow_large=True to read it.'
            )
        resources = self._collect_skill_resources(info, content)
        result = {
            'status': 'ok',
            'name': name,
            'skill_key': info['key'],
            'version': 'latest',
            'path': skill_md,
            'content': content,
            'resources': resources,
        }
        on_load = getattr(self, 'on_skill_loaded', None)
        if on_load is not None:
            result['tool_dependencies'] = on_load(info['key'], info.get('allowed-tools'))
        self._loaded_skills[info['key']] = {
            key: result[key]
            for key in ('status', 'name', 'skill_key', 'version', 'path', 'content', 'resources')
            if key in result
        }
        if 'tool_dependencies' in result:
            self._loaded_skills[info['key']]['tool_dependencies'] = result['tool_dependencies']
        return result

    def read_file(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        info, error = self._get_visible_skill_info(name)
        if error:
            self._raise_skill_lookup_error(name, error)
        if not info:
            raise ToolExecutionError(f'Skill not found: {name}')
        try:
            normalized_rel_path = self._normalize_skill_rel_path(rel_path)
        except ValueError as exc:
            raise ToolExecutionError(
                f'Invalid reference path {rel_path!r} for skill {name}: {exc}'
            ) from exc
        base = info['path']
        path = self._fs_join(base, normalized_rel_path)
        try:
            return {'status': 'ok', 'path': path, 'content': self._fs_read(path)}
        except Exception as e:
            raise ToolExecutionError(
                f'Failed to read reference {normalized_rel_path!r} for skill {name} at {path}: {e}'
            ) from e

    @staticmethod
    def _raise_skill_lookup_error(name: str, error: Dict[str, str]) -> None:
        status = error.get('status')
        if status == 'ambiguous':
            matches = error.get('matches') or []
            raise ToolExecutionError(
                str(error.get('error') or f'Ambiguous skill name {name!r}; matches: {matches}')
            )
        raise ToolExecutionError(f'Skill not found: {name}')

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

    def _raise_run_script_exception(self, name: str, rel_path: str, cwd: Optional[str],
                                    run_cwd: Optional[str], exc: Exception) -> None:
        error_cwd = run_cwd or cwd
        context = f'skill {name}, script {rel_path!r}, cwd {error_cwd!r}'
        if isinstance(exc, ValueError):
            raise ToolExecutionError(
                f'Invalid run_script path argument for skill {name}, script {rel_path!r}, cwd {cwd!r}: {exc}'
            ) from exc
        if isinstance(exc, FileNotFoundError):
            raise ToolExecutionError(f'run_script filesystem path not found for {context}: {exc}') from exc
        if isinstance(exc, subprocess.TimeoutExpired):
            raise ToolExecutionError(
                f'run_script timed out after {exc.timeout} seconds for {context}.'
            ) from exc
        raise ToolExecutionError(f'run_script execution failed for {context}: {exc}') from exc

    def _normalize_script_result(self, result: Dict, skill_info: Optional[Dict] = None) -> Dict:
        if result.get('status') == 'ok' and result.get('exit_code', 0) != 0:
            result['status'] = 'failed'
        if result.get('status') == 'needs_approval':
            raise ToolExecutionError.approval_required(
                str(result.get('reason') or 'Skill script execution requires approval.')
            )
        if result.get('status') in ('error', 'failed', 'missing'):
            reason = str(
                result.get('error') or result.get('stderr')
                or result.get('stdout') or 'Skill script execution failed.'
            )
            exit_code = result.get('exit_code')
            if exit_code is not None:
                reason = f'Skill script execution failed with exit code {exit_code}: {reason}'
            raise self._script_failure_error(reason, result, skill_info)
        return result

    @staticmethod
    def _script_failure_error(
        reason: str, result: Dict, skill_info: Optional[Dict],
    ) -> ToolExecutionError:
        if result.get('status') == 'missing':
            return ToolExecutionError(reason)
        raw_meta = (skill_info or {}).get('raw_meta') or {}
        missing_env = collect_missing_env_hints(
            reason,
            result.get('stderr'),
            result.get('stdout'),
            declared_required=raw_meta.get('required_env'),
        )
        if not missing_env:
            return ToolExecutionError(reason)
        return ToolExecutionError.with_missing_env(
            format_missing_env_message(reason, missing_env),
            missing_env,
        )

    def run_script(self, name: str, rel_path: str, args: Optional[List[str]] = None,
                   cwd: Optional[str] = None) -> Dict[str, str]:
        info, error = self._get_visible_skill_info(name)
        if error:
            self._raise_skill_lookup_error(name, error)
        if not info:
            raise ToolExecutionError(f'Skill not found: {name}')
        try:
            normalized_rel_path = self._normalize_skill_rel_path(rel_path)
        except ValueError as exc:
            raise ToolExecutionError(
                f'Invalid script path {rel_path!r} for skill {name}: {exc}'
            ) from exc
        if not normalized_rel_path.startswith('scripts/'):
            raise ToolExecutionError(
                f'run_script path {normalized_rel_path!r} for skill {name} must be under scripts/.'
            )
        base = info['path']
        temp_dir = None
        run_cwd = None
        try:
            base, temp_dir = self._materialize_script_base(base, normalized_rel_path)
            script_path = self._resolve_local_skill_child(base, normalized_rel_path)
            run_cwd = self._resolve_run_cwd(base, cwd)
            script_exists = os.path.exists(script_path) if temp_dir is not None else self._fs.exists(script_path)
            if not script_exists:
                raise ToolExecutionError(
                    f'Skill script {normalized_rel_path!r} for skill {name} was not found at {script_path}.'
                )
            if self._sandbox is None or not hasattr(self._sandbox, 'execute_script'):
                raise ToolExecutionError(
                    f'The configured sandbox does not support executing skill {name} script '
                    f'{normalized_rel_path!r}.'
                )
            from lazyllm.tools.tool_config_inject import get_dynamic_env_vars
            script_env = dict(get_dynamic_env_vars())
            result = self._sandbox.execute_script(
                source_dir=base,
                rel_path=normalized_rel_path,
                args=args,
                cwd=os.path.relpath(run_cwd, os.path.realpath(os.path.abspath(base))),
                env=script_env,
            )
            return self._normalize_script_result(result, info)
        except ToolExecutionError:
            raise
        except Exception as exc:
            self._raise_run_script_exception(name, normalized_rel_path, cwd, run_cwd, exc)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def read_reference(self, name: str, rel_path: str, **kwargs) -> Dict[str, str]:
        return self.read_file(name=name, rel_path=rel_path, **kwargs)

    def get_skill_tools(self) -> List:
        tools = [
            self._build_search_skill_tool(),
            self._build_get_skill_tool(),
        ]
        if self._skill_tool_mode != SKILL_TOOL_MODE_DISCOVERY:
            tools.extend([
                self._build_read_skill_resource_tool(),
                self._build_run_skill_script_tool(),
            ])
        return tools

    def _build_search_skill_tool(self):
        @fc_register(host_file='NONE', tool_source='skill')
        def search_skill(
            query: str,
            limit: int = DEFAULT_SEARCH_LIMIT,
        ) -> dict:
            '''Find a skill for the current task. This is the only skill-lookup tool.

            If the user asks to search, find, or recommend a skill first, call
            this tool even when a catalog entry looks obvious. Prefer query
            only. Describe the user's task in query. Results are ranked
            candidates, not an exhaustive inventory.
            Do not repeatedly call this tool to enumerate the skill library.
            If the user only asked to find or recommend skills, stop after this
            tool; do not call get_skill just to verify names.

            Args:
                query (str): Task description or skill name. Always required;
                    this is the default and preferred way to search.
                limit (int, optional): Maximum number of candidates. Defaults to 5.
            '''
            return self.search_skill(query=query, limit=limit)
        return search_skill

    def _build_get_skill_tool(self):
        @fc_register(host_file='NONE', tool_source='skill')
        def get_skill(name: str, allow_large: bool = False) -> dict:
            '''Load SKILL.md and the declared resource manifest.

            Catalog descriptions are routing metadata only. Call this before
            executing a skill or reading its resources.
            When the user names a skill, call this directly and never fall back
            to search_skill. Resolution does not consume the search budget.

            Args:
                name (str): Full skill key or unique skill name. Resolution is
                    direct and never consumes the search_skill budget.
                allow_large (bool, optional): Allow loading large SKILL.md. Defaults to False.
            '''
            return self.get_skill(name=name, allow_large=allow_large)
        return get_skill

    def _build_read_skill_resource_tool(self):
        @fc_register(host_file='NONE', tool_source='skill')
        def read_skill_resource(name: str, rel_path: str, **kwargs) -> dict:
            '''Read a resource declared by a loaded skill.

            The skill must already be loaded with get_skill. rel_path must appear
            in that skill's resources manifest.

            Args:
                name (str): Skill key or unique skill name.
                rel_path (str): Resource path copied from the loaded skill.
            '''
            return self._read_loaded_skill_resource(name=name, rel_path=rel_path, **kwargs)
        return read_skill_resource

    def _build_run_skill_script_tool(self):
        @fc_register(host_file='OPAQUE', exclusive=True, tool_source='skill')
        def run_skill_script(name: str, rel_path: str, args: Optional[List[str]] = None,
                             cwd: Optional[str] = None) -> dict:
            '''Run a script declared by a loaded skill.

            The skill must already be loaded with get_skill. rel_path must appear
            in that skill's resources manifest and stay under scripts/.

            Args:
                name (str): Skill key or unique skill name.
                rel_path (str): Script path copied from the loaded skill.
                args (list[str], optional): Script arguments.
                cwd (str, optional): Working directory.
            '''
            return self._run_loaded_skill_script(name=name, rel_path=rel_path, args=args, cwd=cwd)
        return run_skill_script

    def _build_read_reference_tool(self):
        return self._build_read_skill_resource_tool()

    def _build_run_script_tool(self):
        return self._build_run_skill_script_tool()

    def _format_skills_list(self, names: List[str], *, include_location: bool = True) -> str:
        lines = []
        for name in names:
            info = self._skills_index.get(name)
            if not info:
                continue
            desc = self._routing_description(info)
            if include_location:
                lines.append(
                    f'- {name}: {desc} (source: {info.get("source", "file")}, '
                    f'path: {info.get("path")})'
                )
            else:
                lines.append(f'- {name}: {desc}')
        return '\n'.join(lines)

    def _format_skills_locations(self) -> str:
        return '\n'.join(f'- {path}' for path in self._skills_dir)

    def _routing_description(self, info: Dict[str, Any]) -> str:
        meta = info.get('raw_meta') or {}
        preferred = meta.get('when_to_use') or meta.get('when-to-use') or meta.get('trigger')
        if preferred:
            return str(preferred).strip()[:240]
        description = str(info.get('description') or '')
        return self._routing_description_text(description, info.get('name') or '')

    @staticmethod
    def _routing_description_text(description: str, fallback: str = '') -> str:
        text = (description or '').strip()
        if not text:
            return str(fallback or '').strip()[:240]
        parts = _INSTRUCTIONAL_SPLIT.split(text, maxsplit=1)
        cut = parts[0].split('\n', 1)[0].strip(' ，,;；')
        if len(parts) == 1 and len(cut) < 12:
            cut = text.split('\n', 1)[0].strip()
        return cut[:240]

    def _collect_skill_resources(self, info: Dict[str, Any], content: str) -> List[str]:
        resources: List[str] = []
        seen = set()

        def add(rel_path: str) -> None:
            try:
                normalized = self._normalize_skill_rel_path(rel_path)
            except ValueError:
                return
            if normalized not in seen:
                seen.add(normalized)
                resources.append(normalized)

        for folder in ('references', 'scripts'):
            self._walk_skill_folder(info['path'], folder, add)
        for match in re.findall(r'(?:references|scripts)/[A-Za-z0-9._/-]+', content or ''):
            add(match.rstrip('.,);]}`"\''))
        return resources

    def _walk_skill_folder(self, base: str, folder: str, add) -> None:
        stack = [(self._fs_join(base, folder), folder)]
        while stack:
            current, prefix = stack.pop()
            for entry in self._fs_listdir(current):
                name = ''
                is_dir = False
                if isinstance(entry, dict):
                    name = str(entry.get('name') or '').rstrip('/')
                    name = name.rsplit('/', 1)[-1]
                    is_dir = bool(entry.get('type') == 'directory' or entry.get('isdir'))
                else:
                    name = str(entry).rstrip('/').rsplit('/', 1)[-1]
                if not name or name in ('.', '..'):
                    continue
                rel = f'{prefix}/{name}'
                if is_dir:
                    stack.append((self._fs_join(current, name), rel))
                    continue
                add(rel)

    def _loaded_skill_guard(self, name: str, rel_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        info, error = self._get_visible_skill_info(name)
        if error:
            self._raise_skill_lookup_error(name, error)
        if not info:
            return None, {'status': 'error', 'error': 'Skill not found', 'retry': False}
        loaded = self._loaded_skills.get(info['key'])
        if not loaded:
            return None, {'status': 'error', 'error': 'skill_not_loaded', 'retry': False}
        try:
            normalized = self._normalize_skill_rel_path(rel_path)
        except ValueError as exc:
            return None, {'status': 'error', 'error': str(exc), 'retry': False}
        declared = set(loaded.get('resources') or [])
        if normalized not in declared:
            return None, {
                'status': 'error',
                'error': 'resource_not_declared',
                'rel_path': normalized,
                'retry': False,
            }
        return normalized, None

    def _read_loaded_skill_resource(self, name: str, rel_path: str, **kwargs) -> Dict[str, Any]:
        normalized, error = self._loaded_skill_guard(name, rel_path)
        if error:
            return error
        return self.read_file(name=name, rel_path=normalized, **kwargs)

    def _run_loaded_skill_script(
        self,
        name: str,
        rel_path: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized, error = self._loaded_skill_guard(name, rel_path)
        if error:
            return error
        return self.run_script(name=name, rel_path=normalized, args=args, cwd=cwd)


def inherit_skill_scope(
    loadable: Optional[Iterable[str]] = None,
    catalog: Optional[Iterable[str]] = None,
    extra_catalog: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    '''Build child loadable scope and prompt catalog from a parent snapshot.

    The child keeps the parent's loadable/searchable keys. The prompt catalog
    defaults to the parent's catalog, intersected with loadable keys, then
    optionally appends extra catalog entries that remain in scope.
    '''
    loadable_keys = SkillManager._parse_skills(loadable)
    allowed = set(loadable_keys)
    catalog_keys = [key for key in SkillManager._parse_skills(catalog) if key in allowed]
    for key in SkillManager._parse_skills(extra_catalog):
        if key in allowed and key not in catalog_keys:
            catalog_keys.append(key)
    return loadable_keys, catalog_keys
