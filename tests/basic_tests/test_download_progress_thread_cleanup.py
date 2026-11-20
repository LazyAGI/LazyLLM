import os
import queue
import threading
from pathlib import Path

import pytest

from lazyllm.components.utils.downloader import ModelManager


REPO = os.environ.get('LAZYLLM_TEST_DOWNLOAD_REPO')
pytestmark = pytest.mark.skipif(not REPO, reason='Requires an explicit public tokenizer-only download repository')


def test_failed_download_stops_progress_thread(tmp_path):
    blocker = tmp_path / 'cache'
    blocker.write_text('ordinary file blocks directory creation')
    threads = queue.Queue()

    def callback(current, total):
        threads.put(threading.current_thread())

    manager = ModelManager('huggingface', token='', cache_dir=str(blocker), model_path='')
    with pytest.raises(RuntimeError, match='Model download failed'):
        manager.download(REPO, call_back=callback)
    worker = threads.get(timeout=2)
    try:
        worker.join(timeout=1.2)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            args = worker._args
            assert args[3] is callback
            assert str(args[0]).startswith(str(blocker) + os.sep)
            args[2].set()
            worker.join(timeout=2)
            assert not worker.is_alive()


def test_failed_download_without_callback_preserves_error(tmp_path):
    blocker = tmp_path / 'cache'
    blocker.write_text('ordinary file blocks directory creation')
    manager = ModelManager('huggingface', token='', cache_dir=str(blocker), model_path='')
    with pytest.raises(RuntimeError, match='Model download failed'):
        manager.download(REPO)


def test_successful_download_preserves_result_and_stops_thread(tmp_path):
    threads = queue.Queue()

    def callback(current, total):
        threads.put(threading.current_thread())

    manager = ModelManager('huggingface', token='', cache_dir=str(tmp_path / 'cache'), model_path='')
    result = manager.download(REPO, call_back=callback)
    worker = threads.get(timeout=2)
    assert not worker.is_alive()
    assert Path(result).is_dir()
    assert (Path(result) / 'tokenizer.json').is_file()
