import pytest

import lazyllm
from lazyllm.hook import LazyLLMHook
from lazyllm.flow.flow import FlowException


class BaseHook(LazyLLMHook):
    def __init__(self, obj, events, pre_error=False):
        self.events = events
        self.pre_error = pre_error

    def pre_hook(self, *args, **kwargs):
        self.events.append('pre')
        if self.pre_error:
            raise ValueError('pre failure')

    def post_hook(self, output):
        self.events.append('post')

    def on_error(self, error):
        self.events.append('error')


class LegacyHook(BaseHook):
    def report(self):
        self.events.append('report')
        return 'reported'


@pytest.mark.parametrize('phase', ['success', 'work_error', 'pre_error'])
def test_legacy_report_runs_once_through_real_module(phase):
    events = []

    def work(value):
        if phase == 'work_error':
            raise ValueError('work failure')
        return value + 1

    module = lazyllm.ActionModule(work)

    class RegisteredHook(LegacyHook):
        def __init__(self, obj):
            self.pre_error = phase == 'pre_error'

    RegisteredHook.events = events
    module.register_hook(RegisteredHook(module))
    if phase == 'work_error':
        with pytest.raises(FlowException):
            module(1)
        assert events == ['pre', 'error', 'report']
    else:
        assert module(1) == 2
        assert events == (['pre', 'report'] if phase == 'pre_error' else ['pre', 'post', 'report'])


def test_inherited_legacy_report_return_is_preserved():
    class ChildHook(LegacyHook):
        pass

    events = []
    assert ChildHook(None, events).finalize() == 'reported'
    assert events == ['report']


def test_explicit_finalize_takes_precedence():
    class CurrentHook(LegacyHook):
        def __init__(self, obj):
            self.pre_error = False

        def finalize(self):
            self.events.append('finalize')

    events = []
    module = lazyllm.ActionModule(lambda value: value + 1)
    CurrentHook.events = events
    module.register_hook(CurrentHook(module))
    assert module(1) == 2
    assert events == ['pre', 'post', 'finalize']


def test_missing_both_finalizers_keeps_existing_error():
    with pytest.raises(NotImplementedError):
        BaseHook(None, []).finalize()
