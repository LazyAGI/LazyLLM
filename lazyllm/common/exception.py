import sys
import types


_expected_trip_modules = {}
_expected_trip_continuous_modules = {}

def _register_trim_module(expected: dict, continuous: bool = False):
    expected = {k: [v] if isinstance(v, str) else v for k, v in expected.items()}
    if continuous:
        global _expected_trip_continuous_modules
        _expected_trip_continuous_modules.update(expected)
    else:
        global _expected_trip_modules
        _expected_trip_modules.update(expected)

def _is_lazyllm_internal_frame(frame, continuous: bool = False):
    global _expected_trip_modules, _expected_trip_continuous_modules
    expected = _expected_trip_continuous_modules if continuous else _expected_trip_modules
    mod = frame.f_globals.get('__name__', '')
    if not mod.startswith('lazyllm.') and 'lazyllm/components/deploy/relay' not in frame.f_code.co_filename: return False
    if mod in expected and frame.f_code.co_name in expected[mod]: return True
    return False

def _trim_traceback(tb):
    kept, keep_call = [], True
    global _expected_trip_modules, _expected_trip_continuous_modules
    while tb:
        if not _is_lazyllm_internal_frame(tb.tb_frame):
            if _is_lazyllm_internal_frame(tb.tb_frame, continuous=True):
                if keep_call:
                    kept.append(tb)
                    keep_call = False
            else:
                kept.append(tb)
                keep_call = True
        tb = tb.tb_next

    new_tb = None
    for tb in reversed(kept):
        new_tb = types.TracebackType(tb_next=new_tb, tb_frame=tb.tb_frame, tb_lasti=tb.tb_lasti, tb_lineno=tb.tb_lineno)
    return new_tb

_original_excepthook = sys.excepthook

def _lazyllm_excepthook(exc_type, exc_value, tb):
    _original_excepthook(exc_type, exc_value, _trim_traceback(tb))

sys.excepthook = _lazyllm_excepthook


class HandledException(Exception):
    ''' Builtin class. Exception that is intended to be handled exceptions. '''
    pass

def _change_exception_type(e, new_type):
    new_exc = new_type(str(e)).with_traceback(e.__traceback__)
    new_exc.__cause__ = e.__cause__
    new_exc.__context__ = e.__context__
    new_exc.__suppress_context__ = e.__suppress_context__
    return new_exc
