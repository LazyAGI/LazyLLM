import multiprocessing

import pytest

from lazyllm.common.multiprocessing import ForkProcess, SpawnProcess, _ctx


@pytest.mark.parametrize('process_class,original', [(SpawnProcess, 'fork'), (ForkProcess, 'spawn')])
def test_start_failure_restores_method(process_class, original):
    if original not in multiprocessing.get_all_start_methods():
        pytest.skip('Start method unavailable')
    saved_method = multiprocessing.get_start_method()
    parent = multiprocessing.current_process()
    saved_daemon = parent.daemon
    try:
        multiprocessing.set_start_method(original, force=True)
        parent.daemon = True
        with pytest.raises(AssertionError, match='daemonic processes'):
            process_class(target=None).start()
        assert multiprocessing.get_start_method() == original
    finally:
        parent.daemon = saved_daemon
        multiprocessing.set_start_method(saved_method, force=True)


@pytest.mark.parametrize('raises', [False, True])
def test_nested_context_restores_method(raises):
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip('Fork unavailable')
    original = multiprocessing.get_start_method()
    try:
        with _ctx('fork'):
            assert multiprocessing.get_start_method() == 'fork'
            try:
                with _ctx('spawn'):
                    assert multiprocessing.get_start_method() == 'spawn'
                    if raises:
                        raise ValueError('controlled failure')
            except ValueError:
                assert raises
            assert multiprocessing.get_start_method() == 'fork'
        assert multiprocessing.get_start_method() == original
    finally:
        multiprocessing.set_start_method(original, force=True)
