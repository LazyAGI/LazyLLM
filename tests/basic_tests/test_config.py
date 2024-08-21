import lazyllm
from lazyllm.configs import Mode
import os
import copy
import pytest
import contextlib
import inspect


isolate_env = "PARROTS_ISOLATE_STATUS"


def isolated(func):
    def run_subprocess(self, pytester):
        # python_path = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        file_path = inspect.getfile(self.__class__)
        class_name = self.__class__.__name__
        method_name = func.__name__
        test_func = file_path + '::' + class_name + '::' + method_name
        with clear_env():
            with set_env(isolate_env, "IN_SUBPROCESS"):
                result = pytester.runpytest_subprocess(test_func)
        assert result.ret == 0

    if not inspect.isfunction(func):
        raise TypeError("Decorator 'isolated' can only decorate functions.")

    fn_code = func.__code__
    if 'self' not in fn_code.co_varnames or fn_code.co_argcount != 1:
        raise TypeError("Decorated function should be method and "
                        "have exactly one argument 'self'.")

    isolate_status = os.getenv(isolate_env)
    if isolate_status == "IN_SUBPROCESS":
        return func
    # set environ variable to 'OFF' to skip all isolated tests.
    elif isolate_status == "OFF":
        return pytest.mark.skip(func)
    else:
        return pytest.mark.isolate(run_subprocess)


class TestConfig(object):
    def test_refresh(self):
        origin = copy.deepcopy(lazyllm.config.impl)
        os.environ['LAZYLLM_GPU_TYPE'] = 'H100'
        lazyllm.config.refresh('LAZYLLM_GPU_TYPE')
        assert lazyllm.config.impl['gpu_type'] == 'H100'
        os.environ['LAZYLLM_GPU_TYPE'] = origin['gpu_type']
        lazyllm.config.refresh('gpu_type')
        assert lazyllm.config.impl['gpu_type'] == origin['gpu_type']
        lazyllm.config.refresh()
        assert lazyllm.config.impl == origin

    def test_config_mode(self):
        print(os.environ.get('LAZYLLM_DISPLAY'))
        assert lazyllm.config['mode'] == Mode.Normal

    @isolated
    def test_config_disp(self):
        print(os.environ.get('LAZYLLM_DISPLAY'))
        assert lazyllm.config['mode'] == Mode.Display

@contextlib.contextmanager
def clear_env():
    LAZYLLM_DISPLAY = "LAZYLLM_DISPLAY"

    env_list = [
        LAZYLLM_DISPLAY,
    ]
    env_flags = [os.getenv(env) for env in env_list]
    print(env_flags)
    for env, flag in zip(env_list, env_flags):
        if flag is not None:
            os.environ[env] = ""
            if os.getenv(env) is not None:
                del os.environ[env]

    yield

    for env, flag in zip(env_list, env_flags):
        if flag is not None:
            os.environ[env] = flag


@contextlib.contextmanager
def set_env(environ, value):
    assert isinstance(value, str)
    original_value = os.getenv(environ)
    os.environ['LAZYLLM_DISPLAY'] = '1'

    os.environ[environ] = value
    yield

    if original_value is None:
        os.environ.pop(environ)
    else:
        os.environ[environ] = original_value
