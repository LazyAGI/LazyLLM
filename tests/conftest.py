import os, re
import pytest

_GLOBAL_CHANGED_FILES = None
def pytest_runtest_setup(item):

    def get_changed_files():
        global _GLOBAL_CHANGED_FILES
        if _GLOBAL_CHANGED_FILES is not None:
            return _GLOBAL_CHANGED_FILES

        env_str = os.getenv("CHANGED_FILES")
        _GLOBAL_CHANGED_FILES = env_str.split(',') if env_str is not None else []
        return _GLOBAL_CHANGED_FILES

    for mark in item.iter_markers():
        match mark.name:
            case "run_on_change":
                changed = get_changed_files()
                if changed is None: continue
                files = mark.args
                if not any(f in files for f in changed):
                    pytest.skip(f"Skipped: none of {files} were changed.")

            case "run_on_change_regex":
                changed = get_changed_files()
                if changed is None: continue
                file_patterns = mark.args

                def matches_any_pattern(changed_file, patterns):
                    return any(re.fullmatch(pat, changed_file) for pat in patterns)

                if not any(matches_any_pattern(f, file_patterns) for f in changed):
                    pytest.skip(f"Skipped: none of {file_patterns} were changed.")
