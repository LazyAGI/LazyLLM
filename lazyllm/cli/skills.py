import argparse
import os
import shutil
import sys

import lazyllm
from lazyllm.tools.agent.skill_manager import SkillManager


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
        '  lazyllm skills import <path> [--dir DIR] [--names a,b,c] [--overwrite]'
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
        choices=['init', 'list', 'info', 'delete', 'add', 'import'],
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

    args = parser.parse_args(commands)
    manager = _build_manager(args)

    handlers = {
        'init': _handle_init,
        'list': _handle_list,
        'info': _handle_info,
        'add': _handle_add,
        'import': _handle_import,
        'delete': _handle_delete,
    }
    handlers[args.command](manager, args)
