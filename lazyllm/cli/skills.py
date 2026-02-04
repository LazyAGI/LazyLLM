import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import lazyllm
from lazyllm.tools.agent.skill_manager import SkillManager

DEFAULT_SKILL_REPO = 'https://github.com/LazyAGI/LazyLLM'
ATOMGIT_SKILL_REPO = 'https://atomgit.com/LazyAGI/LazyLLM'
GITEE_SKILL_REPO = 'https://gitee.com/lazy-agi/LazyLLM'
SKILL_RELATIVE_PATH = Path('docs/lazyllm-skill')


def _build_manager(args):
    return SkillManager(dir=args.dir)


def _confirm(prompt: str) -> bool:
    try:
        reply = input(prompt).strip().lower()
    except EOFError:
        return False
    return reply in ('y', 'yes')


def _ensure_within_dirs(path: str, roots: list[str]) -> bool:
    path_abs = os.path.abspath(path)
    for root in roots:
        root_abs = os.path.abspath(root)
        try:
            if os.path.commonpath([path_abs, root_abs]) == root_abs:
                return True
        except ValueError:
            continue
    return False


def _parse_name_list(values):
    if not values:
        return []
    names = []
    for item in values:
        if not item:
            continue
        parts = [p.strip() for p in item.split(',') if p.strip()]
        names.extend(parts)
    return list(dict.fromkeys(names))


def _get_roots(manager, dir_value):
    roots = manager._parse_dirs(dir_value) if dir_value else manager._skills_dir
    return roots or []


def _load_skill_meta(manager, folder):
    skill_md = os.path.join(folder, 'SKILL.md')
    if not os.path.isfile(skill_md):
        return None, 'SKILL.md not found at top level.'
    try:
        with open(skill_md, 'r', encoding='utf-8', errors='replace') as fp:
            meta = manager._extract_yaml_meta(fp.read())
    except OSError as exc:
        return None, f'Failed to read SKILL.md ({exc}).'
    if not manager._is_meta_valid(meta or {}):
        return None, 'Invalid SKILL.md metadata.'
    return meta, None


def _remove_existing_skill(meta_name, manager):
    manager._load_skills_index()
    info = manager._skills_index.get(meta_name)
    if not info:
        return None, None
    skill_path = info.get('path')
    if not skill_path:
        return None, f'Invalid skill path for: {meta_name}'
    skill_dir = os.path.dirname(skill_path)
    roots_abs = {os.path.abspath(p) for p in manager._skills_dir}
    if os.path.abspath(skill_dir) in roots_abs:
        candidate_dirs = [os.path.join(root, meta_name) for root in manager._skills_dir]
        candidate_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
        if candidate_dir:
            skill_dir = candidate_dir
        else:
            return None, f'Refusing to delete the skills root directory: {skill_dir}'
    if not _ensure_within_dirs(skill_dir, manager._skills_dir):
        return None, f'Refusing to delete outside skills directories: {skill_dir}'
    if os.path.abspath(skill_dir) in roots_abs:
        return None, f'Refusing to delete the skills root directory: {skill_dir}'
    try:
        shutil.rmtree(skill_dir, ignore_errors=False)
    except OSError as exc:
        return None, f'Failed to remove existing skill \'{meta_name}\': {exc}'
    return skill_dir, None


def _usage_error():
    lazyllm.LOG.error(
        'Usage:\n'
        '  lazyllm skills init\n'
        '  lazyllm skills list\n'
        '  lazyllm skills info <name>\n'
        '  lazyllm skills delete <name>\n'
        '  lazyllm skills add <path> [-n NAME] [--dir DIR]\n'
        '  lazyllm skills import <path> [--dir DIR] [--names a,b,c] [--overwrite]\n'
        '  lazyllm skills install --agent <name> [--project] [--timeout SEC]'
    )
    sys.exit(1)


def _require_name(args):
    if args.name:
        return
    if args.command == 'add':
        lazyllm.LOG.error('Source skill folder is required for add.')
    elif args.command == 'import':
        lazyllm.LOG.error('Source directory is required for import.')
    else:
        lazyllm.LOG.error('Skill name is required for this command.')
    sys.exit(1)


def _handle_init(_manager, _args):
    roots = SkillManager._parse_dirs(lazyllm.config['skills_dir'])
    home_skills = os.path.join(lazyllm.config['home'], 'skills')
    if home_skills not in roots:
        roots.insert(0, home_skills)
    for path in roots:
        os.makedirs(path, exist_ok=True)
    sep = ', '
    lazyllm.LOG.success(f'Initialized skills directories: {sep.join(roots)}')


def _handle_list(manager, _args):
    lazyllm.LOG.info(manager.list_skill())


def _handle_info(manager, args):
    _require_name(args)
    res = manager.get_skill(args.name, allow_large=args.allow_large)
    if res.get('status') != 'ok':
        status = res.get('status', 'error')
        reason = res.get('reason')
        if status == 'missing':
            lazyllm.LOG.error(f'Skill not found: {args.name}')
        elif status == 'too_large':
            limit = res.get('limit')
            extra = f' (limit: {limit} bytes)' if limit else ''
            lazyllm.LOG.error(
                f'SKILL.md is too large for \'{args.name}\'. Use --allow-large to override.{extra}'
            )
        else:
            msg = reason or res
            lazyllm.LOG.error(f'Failed to load skill \'{args.name}\': {msg}')
        sys.exit(1)
    lazyllm.LOG.info(res.get('content', ''))


def _handle_add(manager, args):
    _require_name(args)
    src = os.path.abspath(os.path.expanduser(args.name))
    if not os.path.isdir(src):
        lazyllm.LOG.error(f'Source skill folder not found: {src}')
        sys.exit(1)
    meta, err = _load_skill_meta(manager, src)
    if err:
        lazyllm.LOG.error(f'{err} ({src})')
        sys.exit(1)
    folder_name = args.folder_name or os.path.basename(src.rstrip(os.sep))
    roots = _get_roots(manager, args.dir)
    if not roots:
        lazyllm.LOG.error('No skills directory configured. Run `lazyllm skills init` first or set SKILLS_DIR.')
        sys.exit(1)
    meta_name = meta.get('name') or folder_name
    manager._load_skills_index()
    if meta_name in manager._skills_index:
        lazyllm.LOG.error(f'Skill name \'{meta_name}\' already exists. Use a different name or delete it first.')
        sys.exit(1)
    target_root = roots[0]
    os.makedirs(target_root, exist_ok=True)
    dest = os.path.join(target_root, folder_name)
    if os.path.exists(dest):
        lazyllm.LOG.error(f'Target skill folder already exists: {dest}')
        sys.exit(1)
    shutil.copytree(src, dest)
    lazyllm.LOG.success(f'Added skill \'{folder_name}\' to {dest}')
    lazyllm.LOG.info(f'Installed skills: {meta_name}')


def _handle_import(manager, args):  # noqa: C901
    _require_name(args)
    src_root = os.path.abspath(os.path.expanduser(args.name))
    if not os.path.isdir(src_root):
        lazyllm.LOG.error(f'Source directory not found: {src_root}')
        sys.exit(1)
    roots = _get_roots(manager, args.dir)
    if not roots:
        lazyllm.LOG.error('No skills directory configured. Run `lazyllm skills init` first or set SKILLS_DIR.')
        sys.exit(1)
    target_root = roots[0]
    os.makedirs(target_root, exist_ok=True)

    manager._load_skills_index()
    existing_names = set(manager._skills_index.keys())
    imported = 0
    skipped = 0
    name_filter = set(_parse_name_list(args.names))
    installed = []

    if os.path.isfile(os.path.join(src_root, 'SKILL.md')):
        candidates = [src_root]
    else:
        candidates = [
            os.path.join(src_root, entry)
            for entry in os.listdir(src_root)
            if os.path.isdir(os.path.join(src_root, entry))
        ]

    for folder in candidates:
        meta, err = _load_skill_meta(manager, folder)
        if err:
            lazyllm.LOG.error(f'Skip {folder}: {err}')
            skipped += 1
            continue
        meta_name = meta.get('name')
        folder_name = os.path.basename(folder.rstrip(os.sep))
        if name_filter and meta_name not in name_filter:
            skipped += 1
            continue
        if meta_name in existing_names:
            if not args.overwrite:
                lazyllm.LOG.error(f'Skip {folder}: skill name \'{meta_name}\' already exists.')
                skipped += 1
                continue
            removed_dir, err = _remove_existing_skill(meta_name, manager)
            if err:
                lazyllm.LOG.error(f'Skip {folder}: {err}')
                skipped += 1
                continue
            if removed_dir:
                lazyllm.LOG.info(f'Removed existing skill \'{meta_name}\' at {removed_dir}.')
            existing_names.discard(meta_name)
        dest = os.path.join(target_root, folder_name)
        if os.path.exists(dest):
            if args.overwrite:
                try:
                    shutil.rmtree(dest)
                except OSError as exc:
                    lazyllm.LOG.error(f'Failed to overwrite {dest}: {exc}')
                    skipped += 1
                    continue
            else:
                lazyllm.LOG.error(f'Skip {folder}: target already exists at {dest}.')
                skipped += 1
                continue
        try:
            shutil.copytree(folder, dest)
        except OSError as exc:
            lazyllm.LOG.error(f'Failed to import {folder}: {exc}')
            skipped += 1
            continue
        imported += 1
        existing_names.add(meta_name)
        installed.append(meta_name)

    lazyllm.LOG.success(f'Imported {imported} skill(s) into {target_root}. Skipped {skipped}.')
    if installed:
        sep = ', '
        lazyllm.LOG.info(f'Installed skills: {sep.join(installed)}')


def _handle_install(_manager, args):
    if not args.agent:
        lazyllm.LOG.error('`--agent` is required for install.')
        sys.exit(1)

    if get_agent_config(args.agent) is None:
        agents = ', '.join(get_all_agents())
        lazyllm.LOG.error(f"Unknown agent '{args.agent}'. Available: {agents}")
        sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix='lazyllm-skill-'))
    try:
        _download_repo(tmp_dir, timeout=args.timeout)

        skill_src = tmp_dir / SKILL_RELATIVE_PATH
        if not skill_src.exists():
            lazyllm.LOG.error(f'lazyllm-skill not found at {skill_src}')
            sys.exit(1)

        target_root = get_path(args.agent, args.project)
        target_dir = target_root / 'lazyllm-skill'

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(skill_src, target_dir)
        lazyllm.LOG.success(f'lazyllm-skill installed to: {target_dir}')
    except subprocess.TimeoutExpired:
        lazyllm.LOG.error(f'Install timeout after {args.timeout}s. Try increasing --timeout.')
        sys.exit(1)
    except Exception as exc:
        lazyllm.LOG.error(f'Failed to install lazyllm-skill: {exc}')
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _handle_delete(manager, args):
    _require_name(args)
    manager._load_skills_index()
    info = manager._skills_index.get(args.name)
    if not info:
        lazyllm.LOG.error(f'Skill not found: {args.name}')
        sys.exit(1)

    skill_path = info.get('path')
    if not skill_path:
        lazyllm.LOG.error(f'Invalid skill path for: {args.name}')
        sys.exit(1)

    skill_dir = os.path.dirname(skill_path)
    roots_abs = {os.path.abspath(p) for p in manager._skills_dir}
    if os.path.abspath(skill_dir) in roots_abs:
        candidate_dirs = [os.path.join(root, args.name) for root in manager._skills_dir]
        candidate_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
        if candidate_dir:
            skill_dir = candidate_dir
        else:
            lazyllm.LOG.error(
                f'Skill \'{args.name}\' appears to live at the skills root. '
                'Cannot safely determine its folder to delete.'
            )
            sys.exit(1)
    if not _ensure_within_dirs(skill_dir, manager._skills_dir):
        lazyllm.LOG.error(f'Refusing to delete outside skills directories: {skill_dir}')
        sys.exit(1)
    if os.path.abspath(skill_dir) in {os.path.abspath(p) for p in manager._skills_dir}:
        lazyllm.LOG.error(
            f'Refusing to delete the skills root directory: {skill_dir}. '
            'Move the skill into a dedicated subfolder first.'
        )
        sys.exit(1)

    if not args.yes:
        if not _confirm(f'Delete skill \'{args.name}\' at {skill_dir}? [y/N]: '):
            lazyllm.LOG.info('Cancelled.')
            return

    shutil.rmtree(skill_dir, ignore_errors=False)
    lazyllm.LOG.success(f'Deleted skill \'{args.name}\' at {skill_dir}')


def skills(commands):
    if not commands:
        _usage_error()

    parser = argparse.ArgumentParser(description='lazyllm skills command')
    parser.add_argument(
        'command',
        type=str,
        help='command',
        choices=['init', 'list', 'info', 'delete', 'add', 'import', 'install'],
    )
    parser.add_argument('name', nargs='?', help='skill name or path')
    parser.add_argument('--dir', dest='dir', default=None,
                        help='Skills directory path(s), comma-separated. Defaults to LAZYLLM_SKILLS_DIR.')
    parser.add_argument('--allow-large', action='store_true', default=False,
                        help='Allow loading oversized SKILL.md files.')
    parser.add_argument('--yes', action='store_true', default=False, help='Skip confirmation for delete.')
    parser.add_argument('-n', '--name', dest='folder_name', default=None,
                        help='Target skill folder name (default: source folder name).')
    parser.add_argument('--names', action='append', default=None,
                        help='Comma-separated skill names (from SKILL.md) to import. Can be used multiple times.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing skill folder when importing.')
    parser.add_argument('--agent', default=None,
                        help='Target agent for `skills install` (e.g., cursor, codex, claude).')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Install at project level for `skills install` (default: user level).')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for repository download in `skills install` (default: 300).')

    args = parser.parse_args(commands)
    manager = _build_manager(args)

    handlers = {
        'init': _handle_init,
        'list': _handle_list,
        'info': _handle_info,
        'add': _handle_add,
        'import': _handle_import,
        'install': _handle_install,
        'delete': _handle_delete,
    }
    handlers[args.command](manager, args)


def _clone_repo(repo: str, dst: Path, timeout: int = 300):
    cmd = ['git', 'clone', repo, str(dst)]
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )


def _download_repo(tmp_dir: Path, timeout: int = 300):
    try:
        _clone_repo(DEFAULT_SKILL_REPO, tmp_dir, timeout=timeout)
    except Exception:
        try:
            _clone_repo(ATOMGIT_SKILL_REPO, tmp_dir, timeout=timeout)
        except Exception:
            _clone_repo(GITEE_SKILL_REPO, tmp_dir, timeout=timeout)


@dataclass
class AgentConfig:
    name: str
    display_name: str
    home_dir: Path | None
    project_dir: Path
    supports_home: bool


AGENT_CONFIGS: dict[str, AgentConfig] = {
    'claude': AgentConfig(
        name='claude',
        display_name='Claude Code',
        home_dir=Path.home() / '.claude' / 'skills',
        project_dir=Path('.claude') / 'skills',
        supports_home=True,
    ),
    'opencode': AgentConfig(
        name='opencode',
        display_name='OpenCode CLI',
        home_dir=Path.home() / '.config' / 'opencode' / 'skill',
        project_dir=Path('.opencode') / 'skill',
        supports_home=True,
    ),
    'codex': AgentConfig(
        name='codex',
        display_name='OpenAI Codex',
        home_dir=Path.home() / '.codex' / 'skills',
        project_dir=Path('.codex') / 'skills',
        supports_home=True,
    ),
    'gemini': AgentConfig(
        name='gemini',
        display_name='Gemini CLI',
        home_dir=Path.home() / '.gemini' / 'skills',
        project_dir=Path('.gemini') / 'skills',
        supports_home=True,
    ),
    'copilot': AgentConfig(
        name='copilot',
        display_name='GitHub Copilot',
        home_dir=Path.home() / '.copilot' / 'skills',
        project_dir=Path('.github') / 'skills',
        supports_home=True,
    ),
    'cursor': AgentConfig(
        name='cursor',
        display_name='Cursor',
        home_dir=Path.home() / '.cursor' / 'skills',
        project_dir=Path('.cursor') / 'skills',
        supports_home=True,
    ),
    'windsurf': AgentConfig(
        name='windsurf',
        display_name='Windsurf',
        home_dir=Path.home() / '.codeium' / 'windsurf' / 'skills',
        project_dir=Path('.windsurf') / 'skills',
        supports_home=True,
    ),
    'qwen': AgentConfig(
        name='qwen',
        display_name='Qwen Code',
        home_dir=Path.home() / '.qwen' / 'skills',
        project_dir=Path('.qwen') / 'skills',
        supports_home=True,
    ),
    'antigravity': AgentConfig(
        name='antigravity',
        display_name='Google Antigravity',
        home_dir=Path.home() / '.gemini' / 'antigravity' / 'skills',
        project_dir=Path('.agent') / 'skills',
        supports_home=True,
    ),
    'openhands': AgentConfig(
        name='openhands',
        display_name='OpenHands',
        home_dir=Path.home() / '.openhands' / 'skills',
        project_dir=Path('.openhands') / 'skills',
        supports_home=True,
    ),
    'cline': AgentConfig(
        name='cline',
        display_name='Cline',
        home_dir=Path.home() / '.cline' / 'skills',
        project_dir=Path('.cline') / 'skills',
        supports_home=True,
    ),
    'goose': AgentConfig(
        name='goose',
        display_name='Goose',
        home_dir=Path.home() / '.config' / 'goose' / 'skills',
        project_dir=Path('.goose') / 'skills',
        supports_home=True,
    ),
    'roo': AgentConfig(
        name='roo',
        display_name='Roo Code',
        home_dir=Path.home() / '.roo' / 'skills',
        project_dir=Path('.roo') / 'skills',
        supports_home=True,
    ),
    'kilo': AgentConfig(
        name='kilo',
        display_name='Kilo Code',
        home_dir=Path.home() / '.kilocode' / 'skills',
        project_dir=Path('.kilocode') / 'skills',
        supports_home=True,
    ),
    'trae': AgentConfig(
        name='trae',
        display_name='Trae',
        home_dir=Path.home() / '.trae' / 'skills',
        project_dir=Path('.trae') / 'skills',
        supports_home=True,
    ),
    'droid': AgentConfig(
        name='droid',
        display_name='Droid',
        home_dir=Path.home() / '.factory' / 'skills',
        project_dir=Path('.factory') / 'skills',
        supports_home=True,
    ),
    'clawdbot': AgentConfig(
        name='clawdbot',
        display_name='Clawdbot',
        home_dir=Path.home() / '.clawdbot' / 'skills',
        project_dir=Path('skills'),
        supports_home=True,
    ),
    'kiro-cli': AgentConfig(
        name='kiro-cli',
        display_name='Kiro CLI',
        home_dir=Path.home() / '.kiro' / 'skills',
        project_dir=Path('.kiro') / 'skills',
        supports_home=True,
    ),
    'pi': AgentConfig(
        name='pi',
        display_name='Pi',
        home_dir=Path.home() / '.pi' / 'agent' / 'skills',
        project_dir=Path('.pi') / 'skills',
        supports_home=True,
    ),
    'neovate': AgentConfig(
        name='neovate',
        display_name='Neovate',
        home_dir=Path.home() / '.neovate' / 'skills',
        project_dir=Path('.neovate') / 'skills',
        supports_home=True,
    ),
    'zencoder': AgentConfig(
        name='zencoder',
        display_name='Zencoder',
        home_dir=Path.home() / '.zencoder' / 'skills',
        project_dir=Path('.zencoder') / 'skills',
        supports_home=True,
    ),
    'amp': AgentConfig(
        name='amp',
        display_name='Amp',
        home_dir=Path.home() / '.config' / 'agents' / 'skills',
        project_dir=Path('.agents') / 'skills',
        supports_home=True,
    ),
    'qoder': AgentConfig(
        name='qoder',
        display_name='Qoder',
        home_dir=Path.home() / '.qoder' / 'skills',
        project_dir=Path('.qoder') / 'skills',
        supports_home=True,
    ),
}


def get_agent_config(agent: str) -> AgentConfig | None:
    return AGENT_CONFIGS.get(agent)


def get_all_agents() -> list[str]:
    return list(AGENT_CONFIGS.keys())


def list_all_agents() -> list[str]:
    return get_all_agents()


def get_user_path(agent: str) -> Path:
    config = get_agent_config(agent)
    if config is None:
        raise ValueError(f'Unknown agent: {agent}')

    if not config.supports_home or config.home_dir is None:
        raise ValueError(f"Agent '{agent}' does not support user-level installation")

    return config.home_dir


def get_project_path(agent: str, project_dir: Path | None = None) -> Path:
    config = get_agent_config(agent)
    if config is None:
        raise ValueError(f'Unknown agent: {agent}')

    if project_dir is None:
        project_dir = Path.cwd()

    return (project_dir / config.project_dir).resolve()


def get_path(agent: str, project_level: bool = False, project_dir: Path | None = None) -> Path:
    if project_level:
        return get_project_path(agent, project_dir)
    return get_user_path(agent)
