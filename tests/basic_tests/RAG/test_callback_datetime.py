from datetime import datetime, timedelta, timezone

import pytest

from lazyllm.tools.rag.parsing_service.server import DocumentProcessor


@pytest.mark.parametrize('tz', [None, timezone.utc, timezone(timedelta(hours=8))])
@pytest.mark.parametrize('as_string', [False, True])
def test_callback_due_handles_naive_and_aware_times(tz, as_string):
    impl = object.__new__(DocumentProcessor._Impl)
    now = datetime.now(tz)
    for delta, expected in [(-60, True), (60, False)]:
        value = now + timedelta(seconds=delta)
        assert impl._is_callback_due({'finished_at': value.isoformat() if as_string else value}) is expected


@pytest.mark.parametrize('value', [None, '', 'invalid'])
def test_callback_due_tolerates_missing_or_legacy_time(value):
    impl = object.__new__(DocumentProcessor._Impl)
    assert impl._is_callback_due({'finished_at': value})
