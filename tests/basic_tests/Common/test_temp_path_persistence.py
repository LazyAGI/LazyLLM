import gc
from pathlib import Path

import pytest

from lazyllm import TempPathGenerator


@pytest.mark.parametrize('persist', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_temp_path_lifetime(persist, raises):
    paths = []
    try:
        with TempPathGenerator(['hello', '世界'], suffix='.data', encoding='utf-16', persist=persist) as paths:
            assert [Path(p).read_text(encoding='utf-16') for p in paths] == ['hello', '世界']
            assert all(Path(p).suffix == '.data' for p in paths)
            if raises:
                raise ValueError('body failure')
    except ValueError:
        assert raises
    gc.collect()
    try:
        assert all(Path(p).exists() == persist for p in paths)
    finally:
        if persist:
            for p in paths:
                Path(p).unlink(missing_ok=True)
            if Path(paths[0]).parent.exists():
                Path(paths[0]).parent.rmdir()
