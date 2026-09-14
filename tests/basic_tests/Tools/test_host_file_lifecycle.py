import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel, model_validator

from lazyllm.tools import (
    HostFileAccess,
    HostFileIntent,
    HostFileResolution,
    ToolExecutionDisposition,
    ToolManager,
    ToolRuntimeMetadata,
    fc_register,
)


def call(name='tool', **arguments):
    return {'id': 'provider-id', 'function': {'name': name, 'arguments': arguments}}


def test_declaration_contract_and_legacy_default():
    assert ToolRuntimeMetadata().host_file_access is HostFileAccess.UNDECLARED
    for capability in ('NONE', 'OPAQUE'):
        assert ToolRuntimeMetadata(host_file_access=capability).host_file_access.value == capability
        with pytest.raises(ValueError):
            ToolRuntimeMetadata(host_file_access=capability, host_file_resolver=lambda args: args)
    with pytest.raises(ValueError):
        ToolRuntimeMetadata(host_file_access='DECLARED')
    with pytest.raises(ValueError):
        ToolRuntimeMetadata(host_file_access='typo')
    with pytest.raises(TypeError):
        ToolRuntimeMetadata(host_file_access='DECLARED', host_file_resolver='not-callable')
    with pytest.raises(ValueError):
        HostFileIntent('relative.txt', 'read')
    with pytest.raises(ValueError):
        HostFileIntent('/absolute.txt', 'execute')


def test_prepare_resolves_once_then_executes_original_arguments(tmp_path):
    events = []
    target = tmp_path / 'result.txt'

    def resolve(args):
        events.append(('resolve', args['count']))
        assert type(args['count']) is int
        args['path'] = str((tmp_path / args['path']).resolve())
        return HostFileResolution(args, (HostFileIntent(args['path'], 'write'),))

    @fc_register(host_file_access='DECLARED', host_file_resolver=resolve, execute_in_sandbox=False)
    def tool(path: str, count: int):
        '''Write a count.

        Args:
            path: Destination path.
            count: Count to write.
        '''
        events.append(('execute', path))
        Path(path).write_text(str(count))
        return path

    manager = ToolManager([tool])
    original = call(path='result.txt', count='2')
    prepared = manager.prepare_tool_calls(original, require_host_file_access=True)
    assert events == [('resolve', 2)] and not target.exists()
    assert prepared[0].host_file_access is HostFileAccess.DECLARED
    assert prepared[0].host_files == (HostFileIntent(str(target), 'write'),)
    assert ('file', target) in prepared[0].access.write_keys
    original['function']['arguments']['path'] = 'forged.txt'
    with pytest.raises(TypeError):
        prepared[0].validated_arguments['path'] = 'forged.txt'
    with pytest.raises(FrozenInstanceError):
        prepared[0].tool_name = 'forged'
    batch = manager.execute_prepared(prepared)
    assert events == [('resolve', 2), ('execute', str(target))]
    assert target.read_text() == '2'
    assert batch.records[0].call_id == 'provider-id'
    assert batch.results[0] == {'ok': True, 'value': str(target)}
    assert 'host_file' not in json.dumps(manager.tools_description)


@pytest.mark.parametrize('failure', ['exception', 'wrong_type', 'invalid_arguments'])
def test_host_resolver_failures_never_execute(failure, tmp_path):
    invoked = []

    def resolve(args):
        if failure == 'exception':
            raise OSError('private resolver error')
        if failure == 'wrong_type':
            return args
        return HostFileResolution({'other': 1}, (HostFileIntent(str(tmp_path), 'read'),))

    @fc_register(host_file_access='DECLARED', host_file_resolver=resolve)
    def tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        invoked.append(value)

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls(call(value='x'))
    assert not prepared[0].ready
    batch = manager.execute_prepared(prepared)
    assert batch.records[0].disposition is ToolExecutionDisposition.PREPARATION_FAILED
    assert not batch.results[0]['ok'] and invoked == []
    assert 'private resolver error' not in str(batch.results)


def test_invalid_schema_never_reaches_host_resolver():
    resolved = []

    @fc_register(host_file_access='DECLARED', host_file_resolver=lambda args: resolved.append(args))
    def tool(count: int):
        '''Return a count.

        Args:
            count: Integer count.
        '''
        return count

    prepared = ToolManager([tool]).prepare_tool_calls(call(count='not-an-int'))
    assert not prepared[0].ready and resolved == []


def test_strict_preparation_checks_all_exposed_tools():
    def tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    manager = ToolManager([tool])
    with pytest.raises(ValueError, match='tool'):
        manager.prepare_tool_calls([], require_host_file_access=True)
    assert manager.execute_with_records(call(value='legacy')).results[0]['ok']
    assert len(manager.prepare_tool_calls([], allowed_tool_names=set(), require_host_file_access=True)) == 0


def test_nested_views_are_read_only_and_results_do_not_mutate_batch():
    @fc_register(host_file_access='NONE')
    def tool(data: dict):
        '''Mutate a private input.

        Args:
            data: Nested input.
        '''
        data['items'].append('executed')
        return data

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls(call(data={'items': ['original']}))
    with pytest.raises(TypeError):
        prepared[0].validated_arguments['data']['items'][0] = 'forged'
    first = manager.execute_prepared(prepared)
    assert first.results[0]['value']['items'] == ['original', 'executed']
    assert prepared[0].validated_arguments['data']['items'] == ('original',)
    assert manager.execute_prepared(prepared).results == first.results


def test_admission_barrier_and_rejected_records_keep_original_order():
    effects = []

    @fc_register(host_file_access='NONE')
    def tool(value: str):
        '''Record a value.

        Args:
            value: Input value.
        '''
        effects.append(value)
        return value

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls([call(value=str(i)) for i in range(3)])
    assert not effects
    batch = manager.execute_prepared(prepared, selected_indices=(2, 0))
    assert sorted(effects) == ['0', '2']
    assert [record.index for record in batch.records] == [0, 1, 2]
    assert batch.records[1].disposition is ToolExecutionDisposition.SKIPPED
    assert batch.records[1].reason == 'authorization_rejected'
    assert not batch.results[1]['ok']
    assert [batch.results[i]['value'] for i in (0, 2)] == ['0', '2']
    effects.clear()
    assert len(manager.execute_prepared(prepared, selected_indices=()).records) == 3
    assert not effects
    with pytest.raises(ValueError):
        ToolManager([tool]).execute_prepared(prepared)
    for invalid, error in [((0, 0), ValueError), ((3,), IndexError), ((True,), IndexError)]:
        with pytest.raises(error):
            manager.execute_prepared(prepared, selected_indices=invalid)


def test_concurrent_batches_keep_separate_inputs():
    @fc_register(host_file_access='NONE')
    def tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    manager = ToolManager([tool])
    batches = [manager.prepare_tool_calls(call(value=str(i))) for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(manager.execute_prepared, batches))
    assert [batch.results[0]['value'] for batch in results] == [str(i) for i in range(12)]


def test_host_intents_share_scheduler_conflicts(tmp_path):
    metadata = ToolRuntimeMetadata(
        host_file_access='DECLARED',
        host_file_resolver=lambda args: HostFileResolution(args, ()),
    )
    assert metadata.host_file_access is HostFileAccess.DECLARED
    intent = HostFileIntent(str(tmp_path / 'deleted.txt'), 'delete')
    resolution = HostFileResolution({}, (intent,))
    assert resolution.access.write_keys == frozenset({('file', tmp_path / 'deleted.txt')})
    snapshot = copy.deepcopy(resolution.arguments)
    assert snapshot == {}


def test_execution_does_not_revalidate_or_reresolve(tmp_path):
    def resolve(args):
        return HostFileResolution(args, (HostFileIntent(str(tmp_path), 'read'),))

    @fc_register(host_file_access='DECLARED', host_file_resolver=resolve)
    def tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls(call(value='validated'))

    def fail(*args):
        pytest.fail('execution must reuse preparation')

    manager.all_tools[0]._validate_input = fail
    manager.all_tools[0]._resolve_runtime_access = fail
    assert manager.execute_prepared(prepared).results[0]['value'] == 'validated'


def test_declared_file_access_drives_actual_scheduler(tmp_path):
    events = []

    def resolve(args):
        return HostFileResolution(args, (HostFileIntent(str(tmp_path / 'file'), args['operation']),))

    @fc_register(host_file_access='DECLARED', host_file_resolver=resolve)
    def tool(operation: str):
        '''Record a scheduled access.

        Args:
            operation: File operation.
        '''
        events.append(operation)
        return operation

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls([call(operation='write'), call(operation='read'), call(operation='delete')])
    assert manager._build_execution_segments([item.access for item in prepared]) == [[0], [1], [2]]
    manager.execute_prepared(prepared)
    assert events == ['write', 'read', 'delete']


def test_skill_tools_have_explicit_capabilities(tmp_path):
    from lazyllm.tools.agent import SkillManager

    manager = ToolManager(SkillManager(dir=str(tmp_path)).get_skill_tools())
    declarations = {tool.name: tool.runtime_metadata.host_file_access for tool in manager.all_tools}
    assert declarations == {
        'get_skill': HostFileAccess.NONE,
        'read_reference': HostFileAccess.NONE,
        'run_script': HostFileAccess.OPAQUE,
    }
    manager.prepare_tool_calls([], require_host_file_access=True)


def test_validation_cannot_change_a_resolver_approved_path(tmp_path):
    class Request(BaseModel):
        path: str

        @model_validator(mode='after')
        def transform_path(self):
            self.path += '.validated'
            return self

    effects = []

    def resolve(args):
        return HostFileResolution(args, (HostFileIntent(args['request'].path, 'read'),))

    @fc_register(host_file_access='DECLARED', host_file_resolver=resolve)
    def tool(request: Request):
        '''Read a request path.

        Args:
            request: File request.
        '''
        effects.append(request.path)

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls(call(request={'path': str(tmp_path / 'file')}))
    assert not prepared[0].ready
    assert not manager.execute_prepared(prepared).results[0]['ok']
    assert not effects


def test_resolved_arguments_do_not_reapply_method_input_adapter(tmp_path):
    adaptations = []

    def adapt(args):
        adaptations.append(True)
        return {'path': args['raw_path']}

    def resolve(args):
        return HostFileResolution(args, (HostFileIntent(args['path'], 'read'),))

    class Toolkit:
        __public_apis__ = ['read']
        __tool_input_adapters__ = {'read': adapt}

        @fc_register(host_file_access='DECLARED', host_file_resolver=resolve)
        def read(self, path: str):
            '''Return a path.

            Args:
                path: File path.
            '''
            return path

    manager = ToolManager([Toolkit()])
    prepared = manager.prepare_tool_calls(call('Toolkit_read', raw_path=str(tmp_path)))
    assert prepared[0].ready
    assert manager.execute_prepared(prepared).results[0]['value'] == str(tmp_path)
    assert adaptations == [True]


def test_preparation_failure_results_cannot_poison_the_batch():
    manager = ToolManager([])
    prepared = manager.prepare_tool_calls(call('missing'))
    first = manager.execute_prepared(prepared)
    first.results[0].update(ok=True, value='forged')
    second = manager.execute_prepared(prepared)
    assert not second.results[0]['ok']
    assert second.records[0].disposition is ToolExecutionDisposition.PREPARATION_FAILED


def test_execution_context_wraps_actual_call_and_keeps_original_binding():
    from contextlib import contextmanager
    from contextvars import ContextVar

    current = ContextVar('test_current_call', default=None)
    seen = []

    @fc_register(host_file_access='NONE')
    def tool(value: str):
        '''Return the executing call context.

        Args:
            value: Input value.
        '''
        return current.get(), value

    @contextmanager
    def scope(prepared):
        token = current.set(prepared.index)
        seen.append(('enter', prepared.index))
        try:
            yield
        finally:
            seen.append(('exit', prepared.index))
            current.reset(token)

    manager = ToolManager([tool])
    apply = manager.all_tools[0].apply
    batch = manager.prepare_tool_calls([call(value='a'), call(value='b')])
    results = manager.execute_prepared(batch, execution_context=scope)
    assert [item['value'] for item in results.results] == [(0, 'a'), (1, 'b')]
    assert manager.all_tools[0].apply.__func__ is apply.__func__
    assert manager.all_tools[0].apply.__self__ is apply.__self__
    assert current.get() is None
    assert sorted(seen) == [('enter', 0), ('enter', 1), ('exit', 0), ('exit', 1)]


def test_context_error_prevents_invocation():
    from contextlib import contextmanager

    effects = []

    @fc_register(host_file_access='NONE')
    def tool(value: str):
        '''Record a call.

        Args:
            value: Input value.
        '''
        effects.append(value)

    @contextmanager
    def denied(prepared):
        raise ValueError('denied')
        yield  # pragma: no cover

    manager = ToolManager([tool])
    batch = manager.prepare_tool_calls(call(value='no'))
    result = manager.execute_prepared(batch, execution_context=denied)
    assert not effects and not result.results[0]['ok']


def test_builtin_paths_use_request_working_directory(tmp_path):
    from lazyllm.tools.agent.file_tool import read_file, write_file, delete_file
    from lazyllm.tools.agent.shell_tool import shell_tool
    from lazyllm.tools.agent.todo_tool import todo_write

    manager = ToolManager([read_file, write_file, delete_file, shell_tool, todo_write])
    batch = manager.prepare_tool_calls(
        call('read_file', path='notes.txt'), require_host_file_access=True, working_directory=str(tmp_path))
    assert batch[0].validated_arguments['path'] == str(tmp_path / 'notes.txt')
    assert batch[0].host_files == (HostFileIntent(str(tmp_path / 'notes.txt'), 'read'),)
