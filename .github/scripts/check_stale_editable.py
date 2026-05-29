#!/usr/bin/env python3
'''Check for stale _lazyllm_editable.pth files outside the venv that may redirect
lazyllm imports to a wrong workspace.'''
import os
import sys
import glob

venv_dir = os.environ.get('VENV_DIR', '')
workspace = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
stale = []

for sp in sys.path:
    if not sp or not os.path.isdir(sp):
        continue
    if venv_dir and sp.startswith(venv_dir):
        continue
    for pth in glob.glob(os.path.join(sp, '_lazyllm_editable.pth')):
        with open(pth) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(('import', '#')):
                    if os.path.isabs(line) and line != workspace:
                        stale.append((pth, line))
                    break

if stale:
    print()
    print('=' * 72)
    print('!!! WARNING: Stale _lazyllm_editable.pth detected outside venv !!!')
    print('!!! This may redirect lazyllm imports to a wrong workspace.   !!!')
    print('!!! Delete the following files to fix:                       !!!')
    for pth, ws in stale:
        print(f'!!!   rm {pth}')
        print(f'!!!     (points to: {ws})')
    print('=' * 72)
    print()
else:
    print('OK: no stale _lazyllm_editable.pth found outside venv')
