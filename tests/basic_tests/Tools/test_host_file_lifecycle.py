from lazyllm.tools.agent.tool_runtime import host_file_access
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pydantic import BaseModel, model_validator

from lazyllm.tools import (
    AuthorizationDecision,
    HostFile,
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
    assert ToolRuntimeMetadata(host_file=HostFile.NONE).host_file_access is HostFileAccess.NONE
    assert ToolRuntimeMetadata(host_file=HostFile.OPAQUE).host_file_access is HostFileAccess.OPAQUE

    def resolver(args):
        return HostFileResolution(args, ())
    metadata = ToolRuntimeMetadata(host_file=resolver)
    assert metadata.host_file_access is HostFileAccess.DECLARED
    assert metadata.host_file_resolver is resolver
    with pytest.raises(TypeError):
        ToolRuntimeMetadata(host_file_access='NONE')
    with pytest.raises(TypeError):
        ToolRuntimeMetadata(host_file_resolver=resolver)
    with pytest.raises(ValueError):
        ToolRuntimeMetadata(host_file='DECLARED')
    with pytest.raises(ValueError):
        ToolRuntimeMetadata(host_file='typo')
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

    @fc_register(host_file=resolve, execute_in_sandbox=False)
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
    batch = manager.execute_prepared(prepared, approved_indices=(0,))
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

    @fc_register(host_file=resolve)
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

    @fc_register(host_file=lambda args: resolved.append(args))
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
    assert not manager.execute_with_records(call(value='legacy')).results[0]['ok']
    assert len(manager.prepare_tool_calls([], allowed_tool_names=set(), require_host_file_access=True)) == 0


def test_nested_views_are_read_only_and_results_do_not_mutate_batch():
    @fc_register(host_file='NONE')
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

    @fc_register(host_file='NONE')
    def tool(value: str):
        '''Record a value.

        Args:
            value: Input value.
        '''
        effects.append(value)
        return value

    manager = ToolManager([tool])
    prepared = manager.prepare_tool_calls(
        [call(value=str(i)) for i in range(3)],
        authorization_policy=lambda _: AuthorizationDecision.ASK,
    )
    assert not effects
    batch = manager.execute_prepared(prepared, approved_indices=(2, 0))
    assert sorted(effects) == ['0', '2']
    assert [record.index for record in batch.records] == [0, 1, 2]
    assert batch.records[1].disposition is ToolExecutionDisposition.SKIPPED
    assert batch.records[1].reason == 'approval_required'
    assert not batch.results[1]['ok']
    assert [batch.results[i]['value'] for i in (0, 2)] == ['0', '2']
    effects.clear()
    assert len(manager.execute_prepared(prepared).records) == 3
    assert not effects
    with pytest.raises(ValueError):
        ToolManager([tool]).execute_prepared(prepared)
    for invalid, error in [((0, 0), ValueError), ((3,), IndexError), ((True,), IndexError)]:
        with pytest.raises(error):
            manager.execute_prepared(prepared, approved_indices=invalid)


def test_concurrent_batches_keep_separate_inputs():
    @fc_register(host_file='NONE')
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
    metadata = ToolRuntimeMetadata(host_file=lambda args: HostFileResolution(args, ()))
    assert metadata.host_file_access is HostFileAccess.DECLARED
    intent = HostFileIntent(str(tmp_path / 'deleted.txt'), 'delete')
    resolution = HostFileResolution({}, (intent,))
    assert host_file_access(resolution.files).write_keys == frozenset({('file', tmp_path / 'deleted.txt')})
    snapshot = copy.deepcopy(resolution.arguments)
    assert snapshot == {}


def test_execution_does_not_revalidate_or_reresolve(tmp_path):
    def resolve(args):
        return HostFileResolution(args, (HostFileIntent(str(tmp_path), 'read'),))

    @fc_register(host_file=resolve)
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

    @fc_register(host_file=resolve)
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
    manager.execute_prepared(prepared, approved_indices=(0, 2))
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

    @fc_register(host_file=resolve)
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

        @fc_register(host_file=resolve)
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

    @fc_register(host_file='NONE')
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

    @fc_register(host_file='NONE')
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
    from lazyllm.tools.agent.file_tool import read, write, remove
    from lazyllm.tools.agent.shell_tool import shell
    from lazyllm.tools.agent.todo_tool import todo_write

    manager = ToolManager([read, write, remove, shell, todo_write])
    batch = manager.prepare_tool_calls(
        call('read', path='notes.txt'), require_host_file_access=True, working_directory=str(tmp_path))
    assert batch[0].validated_arguments['path'] == str(tmp_path / 'notes.txt')
    assert batch[0].host_files == (HostFileIntent(str(tmp_path / 'notes.txt'), 'read'),)


def test_single_host_file_registration_entry_derives_internal_capability(tmp_path):
    from lazyllm.tools import HostFile

    def resolve(arguments):
        from lazyllm.tools import resolve_host_path
        path = resolve_host_path(arguments['path'])
        arguments['path'] = path
        return HostFileResolution(arguments, (HostFileIntent(path, 'read'),))

    @fc_register(host_file=HostFile.NONE)
    def none_tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    @fc_register(host_file=HostFile.OPAQUE)
    def opaque_tool(value: str):
        '''Return a value.

        Args:
            value: Input value.
        '''
        return value

    @fc_register(host_file=resolve)
    def declared_tool(path: str):
        '''Return a path.

        Args:
            path: Input path.
        '''
        return path

    manager = ToolManager([none_tool, opaque_tool, declared_tool])
    metadata = {tool.name: tool.runtime_metadata for tool in manager.all_tools}
    assert metadata['none_tool'].host_file_access is HostFileAccess.NONE
    assert metadata['opaque_tool'].host_file_access is HostFileAccess.OPAQUE
    assert metadata['declared_tool'].host_file_access is HostFileAccess.DECLARED
    prepared = manager.prepare_tool_calls(call('declared_tool', path='a.txt'), working_directory=str(tmp_path))
    assert prepared[0].host_files == (HostFileIntent(str(tmp_path / 'a.txt'), 'read'),)


def test_default_authorization_policy_is_fail_closed_without_approval(tmp_path):
    effects = []

    def resolver(operation):
        def resolve(arguments):
            from lazyllm.tools import resolve_host_path
            arguments['path'] = resolve_host_path(arguments['path'])
            return HostFileResolution(arguments, (HostFileIntent(arguments['path'], operation),))
        return resolve

    def make_tool(name, host_file):
        @fc_register(host_file=host_file, execute_in_sandbox=False)
        def tool(path: str):
            '''Record a path.

            Args:
                path: Input path.
            '''
            effects.append((name, path))
            return path
        tool.__name__ = name
        return tool

    manager = ToolManager([
        make_tool('read_tool', resolver('read')),
        make_tool('write_tool', resolver('write')),
        make_tool('opaque_tool', 'OPAQUE'),
        make_tool('none_tool', 'NONE'),
    ])
    prepared = manager.prepare_tool_calls([
        call('read_tool', path='read.txt'),
        call('write_tool', path='write.txt'),
        call('opaque_tool', path='opaque.txt'),
        call('none_tool', path='none.txt'),
    ], working_directory=str(tmp_path))
    assert [item.authorization for item in prepared] == [
        AuthorizationDecision.ALLOW,
        AuthorizationDecision.ASK,
        AuthorizationDecision.ASK,
        AuthorizationDecision.ALLOW,
    ]

    result = manager.execute_prepared(prepared)
    assert [record.reason for record in result.records] == ['', 'approval_required', 'approval_required', '']
    assert sorted(name for name, _ in effects) == ['none_tool', 'read_tool']

    result = manager.execute_prepared(prepared, approved_indices=(1, 2))
    assert all(item['ok'] for item in result.results)
    assert sorted(name for name, _ in effects[-4:]) == sorted(['read_tool', 'write_tool', 'opaque_tool', 'none_tool'])


def test_host_file_resolution_is_prepare_data_not_filesystem_facade():
    for name in ('execution_scope', 'check_path', 'open_read', 'open_write', 'makedirs',
                 'delete', 'rename', 'copy', 'walk', 'listdir'):
        assert not hasattr(HostFileResolution, name)


def test_model_visible_unsafe_flags_are_removed():
    from lazyllm.tools.agent.file_tool import remove, move, write
    from lazyllm.tools.agent.shell_tool import shell

    manager = ToolManager([write, remove, move, shell])
    descriptions = json.dumps(manager.tools_description)
    assert 'allow_unsafe' not in descriptions


def test_registration_rejects_split_host_file_options():
    def tool(value):
        return value

    tool.__name__ = 'split_host_file_tool'
    with pytest.raises(AssertionError):
        fc_register(host_file_access='NONE')(tool)
    with pytest.raises(AssertionError):
        fc_register(host_file_resolver=lambda args: HostFileResolution(args, ()))(tool)


def test_builtin_file_tools_do_not_duplicate_host_file_scheduler_keys():
    from lazyllm.tools.agent.file_tool import (
        remove, ls, mkdir, move, read, grep, write,
    )
    manager = ToolManager([read, ls, grep, mkdir, write, remove, move])
    for tool in manager.all_tools:
        assert tool.runtime_metadata.read_keys is None
        assert tool.runtime_metadata.write_keys is None


def test_trusted_host_selected_indices_replace_default_admission_policy():
    effects = []

    @fc_register(host_file='OPAQUE')
    def opaque(value: str):
        '''Record an opaque call.

        Args:
            value: Input value.
        '''
        effects.append(value)
        return value

    @fc_register(host_file='NONE')
    def denied(value: str):
        '''Record a denied call.

        Args:
            value: Input value.
        '''
        effects.append(value)
        return value

    manager = ToolManager([opaque, denied])
    ask_batch = manager.prepare_tool_calls(call('opaque', value='approved-by-host'))
    result = manager.execute_prepared(ask_batch, selected_indices=(0,))
    assert result.results[0] == {'ok': True, 'value': 'approved-by-host'}

    deny_batch = manager.prepare_tool_calls(
        call('denied', value='blocked'),
        authorization_policy=lambda _: AuthorizationDecision.DENY,
    )
    result = manager.execute_prepared(deny_batch, selected_indices=(0,))
    assert result.results[0] == {'ok': True, 'value': 'blocked'}
    assert effects == ['approved-by-host', 'blocked']
