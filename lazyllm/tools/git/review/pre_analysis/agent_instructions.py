# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from typing import Dict

from .file_context import _SKIP_DIRS, _read_file_head
from .prompt import _FRAMEWORK_GOTCHAS_NOTICE

_AGENT_INSTRUCTION_FILES = [
    'AGENTS.md', 'AGENTS.override.md', 'CLAUDE.md',
    'GEMINI.md', '.cursorrules', 'CONTRIBUTING.md',
]
_AGENT_INSTRUCTIONS_MAX_CHARS = 8000


def _read_agent_instructions(clone_dir: str) -> str:
    parts = []
    for fname in _AGENT_INSTRUCTION_FILES:
        fpath = os.path.join(clone_dir, fname)
        content = _read_file_head(fpath, _AGENT_INSTRUCTIONS_MAX_CHARS)
        if content:
            parts.append(f'### {fname}\n{content}')
    parts.append(_FRAMEWORK_GOTCHAS_NOTICE)
    combined = '\n\n'.join(parts)
    return combined[:_AGENT_INSTRUCTIONS_MAX_CHARS]


def _build_layered_agents_index(clone_dir: str) -> Dict[str, str]:
    '''Scan clone_dir for sub-directory AGENTS.md/CLAUDE.md files (excluding root).
    Returns {relative_dir_path: combined_content} for each directory that has one.
    '''
    index: Dict[str, str] = {}
    for root, dirs, _files in os.walk(clone_dir):
        if os.path.normpath(root) == os.path.normpath(clone_dir):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            continue
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        parts = []
        for fname in _AGENT_INSTRUCTION_FILES:
            fpath = os.path.join(root, fname)
            content = _read_file_head(fpath, 2000)
            if content:
                parts.append(f'### {os.path.relpath(fpath, clone_dir)}\n{content}')
        if parts:
            rel_dir = os.path.relpath(root, clone_dir).replace('\\', '/')
            index[rel_dir] = '\n\n'.join(parts)
    return index


def _get_local_agent_instructions(agents_index: Dict[str, str], file_path: str) -> str:
    '''For a given file path, collect local AGENTS.md content from its directory
    up to (but not including) the root. Closer directories have higher priority (appended last).
    '''
    rel = file_path.replace('\\', '/').lstrip('/')
    parts_path = rel.split('/')
    ancestor_dirs = []
    for i in range(1, len(parts_path)):
        ancestor_dirs.append('/'.join(parts_path[:i]))
    found = []
    for d in ancestor_dirs:
        if d in agents_index:
            found.append(agents_index[d])
    return '\n\n'.join(found)[:3000] if found else ''
