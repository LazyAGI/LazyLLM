import lazyllm
from lazyllm.configs import Mode
import os
import pytest
import contextlib
import binding
import sys
import inspect
import subprocess
import shlex


isolate_env = "PARROTS_ISOLATE_STATUS"
coverage_env = "PARROTS_COVERAGE_TEST"


def isolated(func):
    def run_subprocess(self, testdir):
        python_path = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        file_path = inspect.getfile(self.__class__)
        class_name = self.__class__.__name__
        method_name = func.__name__
        test_func = file_path + '::' + class_name + '::' + method_name

        coverage_flag = os.getenv(coverage_env) == "ON"
        cov_str = "--cov=parrots --cov=torch" if coverage_flag else ""
        with clear_env():
            with set_env(isolate_env, "IN_SUBPROCESS"):
                result = testdir.runpytest_subprocess(cov_str, test_func)
        if coverage_flag:
            subprocess.call(shlex.split("cp .coverage " + python_path +
                            "/.coverage." + method_name))
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


class TestFn_Config(object):
    def test_config_mode(self):
        print(os.environ.get('LAZYLLM_DISPLAY'))
        assert lazyllm.config['mode'] == Mode.Normal

    @isolated
    def test_config_disp(self):
        os.environ['LAZYLLM_DISPLAY'] = '1'
        print(os.environ.get('LAZYLLM_DISPLAY'))
        ret = lazyllm.config['mode']
        print(ret)  #Mode.Normal
        assert ret == Mode.Display



@contextlib.contextmanager
def clear_env():
    LAZYLLM_DISPLAY = "None"

    env_list = [
        LAZYLLM_DISPLAY,
    ]
    env_flags = [os.getenv(env) for env in env_list]
    print(env_flags)
    for env, flag in zip(env_list, env_flags):
        if flag is not None:
            binding.set_env(env, None)
            # TODO(wangzhihong): python's environment manager is not
            #     updated in time, so we have to del it manually
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

    os.environ[environ] = value
    yield

    if original_value is None:
        os.environ.pop(environ)
    else:
        os.environ[environ] = original_value