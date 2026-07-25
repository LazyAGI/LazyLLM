import json
import threading
import time

import lazyllm
import pytest

from lazyllm.tools import ToolManager, fc_register, serial_tool


def _tool_call(name, arguments=None):
    return {
        'function': {
            'name': name,
            'arguments': arguments or {},
        },
    }


def _value(result):
    assert result['ok'] is True
    return result['value']


def test_serial_tool_rejects_invalid_group():
    with pytest.raises(ValueError, match='non-empty'):
        serial_tool(group=' ')
    with pytest.raises(TypeError, match='non-empty string'):
        serial_tool(group=1)


def test_method_serial_group_is_preserved_without_exposing_schema_metadata():
    class Toolkit:
        __public_apis__ = ['mutate']

        @serial_tool(group='state')
        def mutate(self, value: str):
            """Mutate state.

            Args:
                value: New value.
            """
            return value

    manager = ToolManager([Toolkit()])

    assert manager.tools_info['Toolkit_mutate'].serial_group == 'state'
    assert '__lazyllm_serial_tool_group__' not in json.dumps(manager.tools_description)
    assert 'serial_group' not in json.dumps(manager.tools_description)


def test_same_group_runs_in_order_and_failure_does_not_stop_lane():
    class Toolkit:
        __public_apis__ = ['first', 'fail', 'last']

        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.events = []
            self.lock = threading.Lock()

        def _run(self, name):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.events.append(f'{name}:start')
            time.sleep(0.02)
            with self.lock:
                self.events.append(f'{name}:end')
                self.active -= 1
            return name

        @serial_tool(group='state')
        def first(self):
            """Run first."""
            return self._run('first')

        @serial_tool(group='state')
        def fail(self):
            """Fail."""
            self.events.append('fail')
            raise RuntimeError('expected failure')

        @serial_tool(group='state')
        def last(self):
            """Run last."""
            return self._run('last')

    toolkit = Toolkit()
    results = ToolManager([toolkit])([
        _tool_call('Toolkit_first'),
        _tool_call('Toolkit_fail'),
        _tool_call('Toolkit_last'),
    ])

    assert _value(results[0]) == 'first'
    assert results[1]['ok'] is False
    assert _value(results[2]) == 'last'
    assert toolkit.max_active == 1
    assert toolkit.events == [
        'first:start',
        'first:end',
        'fail',
        'last:start',
        'last:end',
    ]


def test_different_groups_and_plain_tools_run_in_parallel():
    class Toolkit:
        __public_apis__ = ['alpha', 'beta', 'plain']

        def __init__(self):
            self.barrier = threading.Barrier(3)

        def _run(self, value):
            self.barrier.wait(timeout=1)
            return value

        @serial_tool(group='alpha')
        def alpha(self):
            """Run alpha."""
            return self._run('alpha')

        @serial_tool(group='beta')
        def beta(self):
            """Run beta."""
            return self._run('beta')

        def plain(self):
            """Run plain."""
            return self._run('plain')

    results = ToolManager([Toolkit()])([
        _tool_call('Toolkit_alpha'),
        _tool_call('Toolkit_plain'),
        _tool_call('Toolkit_beta'),
    ])

    assert [_value(result) for result in results] == ['alpha', 'plain', 'beta']


def test_serial_sandbox_calls_preserve_results():
    @serial_tool(group='sandbox-state')
    def sandbox_first(value: str):
        """Return a value.

        Args:
            value: Input value.
        """
        return value

    @serial_tool(group='sandbox-state')
    def sandbox_second(value: str):
        """Return a value.

        Args:
            value: Input value.
        """
        return value

    class RecordingSandbox:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            return {'sandbox_call': self.calls, 'has_code': bool(kwargs.get('code'))}

    sandbox = RecordingSandbox()
    results = ToolManager(
        [sandbox_first, sandbox_second],
        sandbox=sandbox,
    )([
        _tool_call('sandbox_first', {'value': 'first'}),
        _tool_call('sandbox_second', {'value': 'second'}),
    ])

    assert results == [
        {'sandbox_call': 1, 'has_code': True},
        {'sandbox_call': 2, 'has_code': True},
    ]


def test_registered_functions_preserve_group_in_both_decorator_orders():
    def inner_first(value: str):
        """Return a value.

        Args:
            value: Input value.
        """
        return value

    def outer_first(value: str):
        """Return a value.

        Args:
            value: Input value.
        """
        return value

    inner_first.__name__ = 'serial_inner_first_test_tool'
    outer_first.__name__ = 'serial_outer_first_test_tool'
    fc_register('tool', execute_in_sandbox=False)(
        serial_tool(group='inner')(inner_first),
    )
    serial_tool(group='outer')(
        fc_register('tool', execute_in_sandbox=False)(outer_first),
    )
    try:
        manager = ToolManager([
            'serial_inner_first_test_tool',
            'serial_outer_first_test_tool',
        ])
        assert manager.tools_info['serial_inner_first_test_tool'].serial_group == 'inner'
        assert manager.tools_info['serial_outer_first_test_tool'].serial_group == 'outer'
    finally:
        lazyllm.tool.remove('serial_inner_first_test_tool')
        lazyllm.tool.remove('serial_outer_first_test_tool')
