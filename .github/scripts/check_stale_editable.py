#!/usr/bin/env python3
'''Check for stale editable install files outside the venv that may redirect
lazyllm imports to a wrong workspace, and remove them automatically.'''
import os
import sys
import glob

venv_dir = os.environ.get('VENV_DIR', '')
workspace = os.environ.get('GITHUB_WORKSPACE', os.getcwd())
stale = []

# Patterns covering both legacy and modern editable install markers
PTH_PATTERNS = ['_lazyllm_editable.pth', '__editable__.lazyllm*.pth', 'lazyllm.egg-link']

for sp in sys.path:
    if not sp or not os.path.isdir(sp):
        continue
    if venv_dir and sp.startswith(venv_dir):
        continue
    for pattern in PTH_PATTERNS:
        for pth in glob.glob(os.path.join(sp, pattern)):
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
    print('WARNING: Stale editable install files detected outside venv.')
    print('These may redirect lazyllm imports to a wrong workspace.')
    print('Removing them now...')
    for pth, ws in stale:
        print(f'  Removing: {pth}')
        print(f'    (was pointing to: {ws})')
        try:
            os.remove(pth)
            print(f'  Removed: {pth}')
        except OSError as e:
            print(f'  ERROR removing {pth}: {e}')
    print('=' * 72)
    print()
else:
    print('OK: no stale editable install files found outside venv')
