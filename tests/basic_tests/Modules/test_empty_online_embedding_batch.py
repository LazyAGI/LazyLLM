import subprocess
import sys

import pytest

from lazyllm.module.llms.onlinemodule.base.onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase


@pytest.mark.parametrize('workers', [1, 2])
def test_empty_public_batch_finishes_without_network(workers):
    code = '''import sys
from lazyllm.module.llms.onlinemodule.base.onlineEmbeddingModuleBase import OnlineEmbeddingModuleBase
def guard(event, args):
    if event == 'socket.connect':
        raise AssertionError('empty batch must not connect to a service')
sys.addaudithook(guard)
module = OnlineEmbeddingModuleBase('http://127.0.0.1:1/embeddings', '', 'text-embedding-ada-002',
                                   skip_auth=True, num_worker=WORKERS)
assert module([]) == []
'''.replace('WORKERS', str(workers))
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.communicate(timeout=10)
        pytest.fail('empty batch did not finish; owned child terminated and waited')
    assert process.returncode == 0, stdout + stderr


def test_nonempty_batch_encapsulation_keeps_order_and_model():
    module = OnlineEmbeddingModuleBase('http://127.0.0.1:1/embeddings', '', 'text-embedding-ada-002',
                                       skip_auth=True, batch_size=2)
    assert module._encapsulated_data(['a', 'b', 'c']) == [
        {'input': ['a', 'b'], 'model': 'text-embedding-ada-002'},
        {'input': ['c'], 'model': 'text-embedding-ada-002'},
    ]
