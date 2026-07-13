import os

import lazyllm
from lazyllm.common.utils import dump_obj, dump_obj_to_file, load_obj
from lazyllm.components.deploy.relay.base import RelayServer


def test_dump_obj_to_file_round_trip_and_cleanup(tmp_path):
    path = tmp_path / 'payload.pkl'
    reference = dump_obj_to_file({'answer': 42}, str(path))

    assert load_obj(reference) == {'answer': 42}
    assert not path.exists()

    none_path = tmp_path / 'none.pkl'
    none_reference = dump_obj_to_file(None, str(none_path))
    assert load_obj(none_reference) is None
    assert not none_path.exists()


def test_relay_file_args_use_lazyllm_temp_dir(tmp_path):
    relay = RelayServer(func=lambda value: value, pass_args_by_file=True)

    with lazyllm.config.temp('temp_dir', str(tmp_path)):
        reference = relay._prepare_obj_arg(dump_obj({'answer': 42}), force=True)

    path = reference.removeprefix('@file:')
    assert os.path.dirname(path) == str(tmp_path)
    assert load_obj(reference) == {'answer': 42}
    assert not os.path.exists(path)


def test_relay_windows_command_quotes_paths_with_spaces(monkeypatch):
    monkeypatch.setattr(os, 'name', 'nt')

    command = RelayServer._join_command([
        r'C:\Program Files\Python\python.exe',
        r'C:\Lazy Mind\relay server.py',
        r'--function=@file:C:\Lazy Mind\payload.pkl',
    ])

    assert command.startswith('"C:\\Program Files\\Python\\python.exe"')
    assert '"C:\\Lazy Mind\\relay server.py"' in command
    assert '"--function=@file:C:\\Lazy Mind\\payload.pkl"' in command
