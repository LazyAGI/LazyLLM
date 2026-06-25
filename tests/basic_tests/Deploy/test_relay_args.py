import importlib.util
import shlex
import subprocess
import sys
import uuid

import pytest

import lazyllm
from lazyllm.components.deploy.relay import base as relay_base


def _identity(value):
    return value


def _load_relay_server_module(monkeypatch, argv):
    module_name = f'_lazyllm_relay_server_test_{uuid.uuid4().hex}'
    server_path = relay_base.os.path.join(
        relay_base.os.path.dirname(relay_base.os.path.abspath(relay_base.__file__)),
        'server.py',
    )
    monkeypatch.setattr(sys, 'argv', [server_path] + argv)
    old_sys_path = list(sys.path)
    old_relay_services = None
    has_relay_services = hasattr(lazyllm.FastapiApp, '__relay_services__')
    if has_relay_services:
        old_relay_services = lazyllm.FastapiApp.__relay_services__.copy()
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = old_sys_path
        if has_relay_services:
            lazyllm.FastapiApp.__relay_services__ = old_relay_services
    return module


def _get_arg_value(tokens, name):
    needle = f'--{name}='
    for token in tokens:
        if token.startswith(needle):
            return token[len(needle):]
    return None


def _assert_file_payload(path, expected_payload):
    assert relay_base.os.path.exists(path), f'{path} not exists'
    with open(path, encoding='utf-8') as f:
        assert f.read() == expected_payload


def test_relay_cmd_writes_serialized_payloads_to_files(monkeypatch):
    monkeypatch.setattr(relay_base, '_should_write_relay_arg_file', lambda: True)
    server = relay_base.RelayServer(
        port=32123,
        func=_identity,
        pre_func=_identity,
        post_func=_identity,
        pythonpath='C:\\path&with spaces',
        security_key='token&with spaces',
        defined_pos='file: "demo.py", line 7',
    )

    cmd = server.cmd().cmd()
    cmd_parts = shlex.split(cmd)

    function_file = _get_arg_value(cmd_parts, 'function_file')
    before_file = _get_arg_value(cmd_parts, 'before_function_file')
    after_file = _get_arg_value(cmd_parts, 'after_function_file')
    pythonpath_file = _get_arg_value(cmd_parts, 'pythonpath_file')
    security_key_file = _get_arg_value(cmd_parts, 'security_key_file')
    defined_pos_file = _get_arg_value(cmd_parts, 'defined_pos_file')
    try:
        assert function_file is not None
        assert before_file is not None
        assert after_file is not None
        assert pythonpath_file is not None
        assert security_key_file is not None
        assert defined_pos_file is not None
        _assert_file_payload(function_file, lazyllm.dump_obj(_identity))
        _assert_file_payload(before_file, lazyllm.dump_obj(_identity))
        _assert_file_payload(after_file, lazyllm.dump_obj(_identity))
        _assert_file_payload(pythonpath_file, 'C:\\path&with spaces')
        _assert_file_payload(security_key_file, 'token&with spaces')
        assert _get_arg_value(cmd_parts, 'function') is None
        assert _get_arg_value(cmd_parts, 'before_function') is None
        assert _get_arg_value(cmd_parts, 'after_function') is None
        assert _get_arg_value(cmd_parts, 'pythonpath') is None
        assert _get_arg_value(cmd_parts, 'security_key') is None
        assert _get_arg_value(cmd_parts, 'defined_pos') is None
        assert relay_base.os.path.exists(defined_pos_file)
        with open(defined_pos_file, encoding='utf-8') as f:
            assert f.read() in {
                lazyllm.dump_obj('file: "demo.py", line 7'),
                lazyllm.dump_obj('file: \\"demo.py\\", line 7'),
            }
    finally:
        for path in (function_file, before_file, after_file, pythonpath_file, security_key_file, defined_pos_file):
            if path:
                try:
                    relay_base.os.unlink(path)
                except FileNotFoundError:
                    pass


def test_relay_payload_file_arg_uses_windows_quoting(monkeypatch, tmp_path):
    temp_dir = tmp_path / 'relay payloads'
    temp_dir.mkdir()
    monkeypatch.setattr(relay_base.os, 'name', 'nt')
    monkeypatch.setattr(relay_base.tempfile, 'tempdir', str(temp_dir))

    arg = relay_base._relay_payload_arg('function', 'payload', use_file=True)
    raw_path = arg.removeprefix('--function_file=').strip()
    assert raw_path.startswith('"')
    assert raw_path.endswith('"')
    assert "'" not in raw_path
    payload_file = raw_path.strip('"')
    try:
        assert raw_path == subprocess.list2cmdline([payload_file])
        _assert_file_payload(payload_file, 'payload')
    finally:
        relay_base.os.unlink(payload_file)


def test_relay_server_reads_payload_from_file(monkeypatch, tmp_path):
    payload = lazyllm.dump_obj(_identity)
    payload_file = tmp_path / 'function.b64'
    payload_file.write_text(payload, encoding='utf-8')

    module = _load_relay_server_module(
        monkeypatch,
        ['--open_port=32124', f'--function_file={payload_file}'],
    )

    assert module.func('ok') == 'ok'
    assert not payload_file.exists()


def test_relay_server_reads_security_key_from_file(monkeypatch, tmp_path):
    payload = lazyllm.dump_obj(_identity)
    payload_file = tmp_path / 'function.b64'
    security_file = tmp_path / 'security.txt'
    payload_file.write_text(payload, encoding='utf-8')
    security_file.write_text('token&value', encoding='utf-8')

    module = _load_relay_server_module(
        monkeypatch,
        ['--open_port=32128', f'--function_file={payload_file}', f'--security_key_file={security_file}'],
    )

    assert module.security_key_arg == 'token&value'
    assert not payload_file.exists()
    assert not security_file.exists()


def test_relay_server_accepts_direct_payload(monkeypatch):
    payload = lazyllm.dump_obj(_identity)

    module = _load_relay_server_module(
        monkeypatch,
        ['--open_port=32125', f'--function={payload}'],
    )

    assert module.func('ok') == 'ok'


def test_relay_server_rejects_missing_function_payload(monkeypatch):
    with pytest.raises(SystemExit):
        _load_relay_server_module(monkeypatch, ['--open_port=32126'])


def test_relay_server_rejects_ambiguous_function_payload(monkeypatch, tmp_path):
    payload = lazyllm.dump_obj(_identity)
    payload_file = tmp_path / 'function.b64'
    payload_file.write_text(payload, encoding='utf-8')

    with pytest.raises(SystemExit):
        _load_relay_server_module(
            monkeypatch,
            ['--open_port=32127', f'--function={payload}', f'--function_file={payload_file}'],
        )
