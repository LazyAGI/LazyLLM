#!/usr/bin/env python3
"""Generate _lazyllm_editable.{pth,py} in a venv's site-packages to override stale system ones.

Usage: python gen_editable.py <venv_site_packages_dir>
"""
import os
import sys


def discover_modules(package_dir):
    """Walk package_dir and map module_name -> absolute_path."""
    mapping = {}
    package_dir = os.path.abspath(package_dir)
    parent = os.path.dirname(package_dir)
    pkg_name = os.path.basename(package_dir)

    for root, dirs, files in os.walk(package_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, parent)
            # Convert filesystem path to module name
            mod = rel.replace(os.sep, '.')
            if mod.endswith('.py'):
                mod = mod[:-3]
            mapping[mod] = fpath

    return mapping


def generate(site_dir, workspace_dir, package_dir):
    known = discover_modules(package_dir)

    pth_content = f"import _lazyllm_editable\n{workspace_dir}\n"

    # Build install() arguments
    source_entries = []
    for mod, path in sorted(known.items()):
        source_entries.append(f"        '{mod}': '{path}'")
    source_str = ',\n'.join(source_entries)

    new_install = f"install({{\n{source_str}\n    }}, {{}}, None, False, True, [], [], '')"

    # Copy template from an existing _lazyllm_editable.py and replace last install() call
    candidates = [
        '/home/mnt/yuezihao/.venvs/lazyllm-overlay/lib/python3.10/site-packages/_lazyllm_editable.py',
        '/mnt/lustre/share_data/env/miniconda3/lib/python3.10/site-packages/_lazyllm_editable.py',
    ]
    template = None
    for c in candidates:
        if os.path.exists(c):
            with open(c) as f:
                template = f.read()
            break
    if template is None:
        print("ERROR: cannot find _lazyllm_editable.py template")
        sys.exit(1)

    # Replace the LAST install() call (the module-level one, not the def)
    last_pos = template.rfind('install(')
    # Find matching closing paren
    depth = 1
    i = last_pos + len('install(')
    while i < len(template) and depth > 0:
        if template[i] == '(':
            depth += 1
        elif template[i] == ')':
            depth -= 1
        i += 1
    new_content = template[:last_pos] + new_install + template[i:]

    # Write files
    pth_path = os.path.join(site_dir, '_lazyllm_editable.pth')
    py_path = os.path.join(site_dir, '_lazyllm_editable.py')

    with open(pth_path, 'w') as f:
        f.write(pth_content)
    with open(py_path, 'w') as f:
        f.write(new_content)

    print(f'Wrote {pth_path}')
    print(f'Wrote {py_path}')
    print(f'Modules mapped: {len(known)}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <venv_site_packages_dir>')
        sys.exit(1)

    site_dir = sys.argv[1]
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(workspace_dir, 'lazyllm')

    generate(site_dir, workspace_dir, package_dir)
