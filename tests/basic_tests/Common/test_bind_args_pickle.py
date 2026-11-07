import pickle
from types import SimpleNamespace

import pytest

import lazyllm
from lazyllm.common.bind import Bind


def collect(*args):
    return args


@pytest.mark.parametrize('unpack', [False, True])
@pytest.mark.parametrize('selection', ['whole', 'item', 'attribute'])
def test_pickle_preserves_unpack_and_repeated_selection(unpack, selection):
    value = lazyllm.package(2, 3)
    reference = Bind.Args('source', unpack=unpack)
    if selection == 'item':
        reference = reference['value']
        input_value = {'value': value}
    elif selection == 'attribute':
        reference = reference.value
        input_value = SimpleNamespace(value=value)
    else:
        input_value = value
    restored = pickle.loads(pickle.dumps(reference))
    for _ in range(2):
        result = restored.get_arg({'source': 'source', 'input': input_value})
        assert result == value
        assert isinstance(result, Bind.Args.Unpack) == unpack


@pytest.mark.parametrize('unpack', [False, True])
def test_actual_process_pool_consumes_serialized_binding(unpack):
    value = lazyllm.package(2, 3)
    reference = Bind.Args('source', unpack=unpack)
    bound = lazyllm.bind(collect, reference)
    with lazyllm.ProcessPoolExecutor(max_workers=1) as executor:
        result = executor.submit(bound, _bind_args_source={'source': 'source', 'input': value}).result(timeout=10)
    assert result == ((2, 3) if unpack else (value,))


def test_actual_parallel_processes_pipeline_output_binding():
    with lazyllm.pipeline(save_result=True) as flow:
        flow.values = lambda value: lazyllm.package(value, value + 1)
        flow.results = lazyllm.parallel(
            lazyllm.bind(collect, flow.output(flow.values, unpack=True)),
            lazyllm.bind(collect, flow.output(flow.values, unpack=False)),
            multiprocessing=True,
            _concurrent=2,
        )
    assert flow(2) == ((2, 3), ((2, 3),))
