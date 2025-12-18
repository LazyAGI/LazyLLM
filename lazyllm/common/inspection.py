import os
import inspect

def _get_callsite(depth: int = 1):
    try:
        frame = inspect.currentframe()
        for _ in range(depth): frame = frame.f_back
        if frame is None: return None
        else:
            while frame.f_code.co_name == '__setattr__' and frame.f_globals.get('__name__', '') == 'lazyllm.common.bind':
                frame = frame.f_back
        return f'"file: {os.path.abspath(frame.f_code.co_filename)}", line {frame.f_lineno}'
    except Exception:
        return None
