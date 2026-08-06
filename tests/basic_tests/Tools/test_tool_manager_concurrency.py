import json
import threading
import time

import lazyllm
import pytest

from lazyllm.flow.flow import FlowException
from lazyllm.tools import ToolManager, fc_register, tool_concurrency
from lazyllm.tools.agent.file_tool import (
    delete_file,
    list_dir,
    make_dir,
    move_file,
    read_file,
    search_in_files,
    write_file,
)


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


def test_concurrency_metadata_is_preserved_and_hidden_from_schema():
    class Toolkit:
        __public_apis__ = ['read']

        @tool_concurrency(read_keys=lambda args: ('file', args['path']))
        def read(self, path: str):
            '''Read a path.

            Args:
                path: File path.
            '''
            return path

    registered_name = 'tool_concurrency_registered_test_tool'

    @tool_concurrency(write_keys=lambda args: ('file', args['path']))
    def registered(path: str):
        '''Write a path.

        Args:
            path: File path.
        '''
        return path

    registered.__name__ = registered_name
    fc_register('tool', execute_in_sandbox=False)(registered)
    try:
        manager = ToolManager([Toolkit(), registered_name])
        assert manager.tools_info['Toolkit_read'].concurrency_spec is not None
        assert manager.tools_info[registered_name].concurrency_spec is not None
        schema = json.dumps(manager.tools_description)
        assert '__lazyllm_tool_concurrency__' not in schema
        assert 'concurrency_spec' not in schema

        builtin_manager = ToolManager([
            read_file, list_dir, search_in_files, make_dir, write_file, delete_file, move_file,
        ])
        assert all(tool.concurrency_spec is not None for tool in builtin_manager.all_tools)
    finally:
        lazyllm.tool.remove(registered_name)


@pytest.mark.parametrize(
    'first_name,first_path,second_name,second_path,expected_max_active',
    [
        ('read', '/tmp/tool-concurrency/a', 'read', '/tmp/tool-concurrency/a', 2),
        ('read', '/tmp/tool-concurrency/a', 'write', '/tmp/tool-concurrency/a', 1),
        ('write', '/tmp/tool-concurrency/a', 'write', '/tmp/tool-concurrency/a/child', 1),
        ('write', '/tmp/tool-concurrency/a', 'write', '/tmp/tool-concurrency/b', 2),
    ],
)
def test_dynamic_file_keys_control_parallelism(
        first_name, first_path, second_name, second_path, expected_max_active):
    class Toolkit:
        __public_apis__ = ['read', 'write']

        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def _run(self, path):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return path

        @tool_concurrency(read_keys=lambda args: ('file', args['path']))
        def read(self, path: str):
            '''Read a path.

            Args:
                path: File path.
            '''
            return self._run(path)

        @tool_concurrency(write_keys=lambda args: ('file', args['path']))
        def write(self, path: str):
            '''Write a path.

            Args:
                path: File path.
            '''
            return self._run(path)

    toolkit = Toolkit()
    results = ToolManager([toolkit])([
        _tool_call(f'Toolkit_{first_name}', {'path': first_path}),
        _tool_call(f'Toolkit_{second_name}', {'path': second_path}),
    ])

    assert [_value(result) for result in results] == [first_path, second_path]
    assert toolkit.max_active == expected_max_active


def test_ordered_segments_exclusive_barrier_and_result_order():
    class Toolkit:
        __public_apis__ = ['write', 'exclusive', 'plain']

        def __init__(self):
            self.events = []
            self.lock = threading.Lock()

        def _run(self, label, delay=0.01):
            with self.lock:
                self.events.append(f'{label}:start')
            time.sleep(delay)
            with self.lock:
                self.events.append(f'{label}:end')
            return label

        @tool_concurrency(write_keys=lambda args: ('file', args['path']))
        def write(self, path: str, label: str, delay: float = 0.01):
            '''Write a path.

            Args:
                path: File path.
                label: Event label.
                delay: Execution delay.
            '''
            return self._run(label, delay)

        @tool_concurrency(exclusive=True)
        def exclusive(self, label: str):
            '''Run exclusively.

            Args:
                label: Event label.
            '''
            return self._run(label)

        def plain(self, label: str):
            '''Run without resource declarations.

            Args:
                label: Event label.
            '''
            return self._run(label)

    toolkit = Toolkit()
    results = ToolManager([toolkit])([
        _tool_call('Toolkit_write', {'path': '/tmp/tool-concurrency/a', 'label': 'first', 'delay': 0.02}),
        _tool_call('Toolkit_write', {'path': '/tmp/tool-concurrency/b', 'label': 'peer', 'delay': 0.04}),
        _tool_call('Toolkit_write', {'path': '/tmp/tool-concurrency/a', 'label': 'second'}),
        _tool_call('Toolkit_exclusive', {'label': 'exclusive'}),
        _tool_call('Toolkit_plain', {'label': 'plain'}),
    ])

    assert [_value(result) for result in results] == ['first', 'peer', 'second', 'exclusive', 'plain']
    assert toolkit.events.index('peer:end') < toolkit.events.index('second:start')
    assert toolkit.events.index('second:end') < toolkit.events.index('exclusive:start')
    assert toolkit.events.index('exclusive:end') < toolkit.events.index('plain:start')


def test_tool_failure_does_not_stop_later_conflicting_calls():
    events = []

    @tool_concurrency(write_keys=lambda args: ('file', args['path']))
    def mutate(path: str, label: str, fail: bool = False):
        '''Mutate a path.

        Args:
            path: File path.
            label: Event label.
            fail: Whether to fail.
        '''
        events.append(label)
        if fail:
            raise RuntimeError('expected failure')
        return label

    results = ToolManager([mutate])([
        _tool_call('mutate', {'path': '/tmp/tool-concurrency/a', 'label': 'first'}),
        _tool_call('mutate', {'path': '/tmp/tool-concurrency/a', 'label': 'fail', 'fail': True}),
        _tool_call('mutate', {'path': '/tmp/tool-concurrency/a', 'label': 'last'}),
    ])

    assert events == ['first', 'fail', 'last']
    assert _value(results[0]) == 'first'
    assert results[1]['ok'] is False
    assert _value(results[2]) == 'last'


def test_sandbox_failures_attempt_all_calls_and_rethrow_first_exception():
    @tool_concurrency(write_keys=lambda args: ('file', args['path']))
    def sandbox_tool(path: str):
        '''Process a path.

        Args:
            path: File path.
        '''
        return path

    class RecordingSandbox:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls in (2, 3):
                raise RuntimeError(f'expected sandbox failure {self.calls}')
            return {'sandbox_call': self.calls, 'has_code': bool(kwargs.get('code'))}

    sandbox = RecordingSandbox()
    manager = ToolManager([sandbox_tool], sandbox=sandbox)
    with pytest.raises(FlowException, match='expected sandbox failure 2'):
        manager([
            _tool_call('sandbox_tool', {'path': '/tmp/tool-concurrency/a'}),
            _tool_call('sandbox_tool', {'path': '/tmp/tool-concurrency/a'}),
            _tool_call('sandbox_tool', {'path': '/tmp/tool-concurrency/a'}),
        ])

    assert sandbox.calls == 3
