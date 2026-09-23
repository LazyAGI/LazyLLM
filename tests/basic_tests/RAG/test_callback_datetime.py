from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy

from lazyllm.tools.rag.parsing_service.base import FINISHED_TASK_QUEUE_TABLE_INFO
from lazyllm.tools.rag.parsing_service.queue import _SQLBasedQueue
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


def test_queue_peek_preserves_stored_time_instead_of_driver_timezone(tmp_path):
    queue = _SQLBasedQueue(
        table_name='finished_time_test', columns=FINISHED_TASK_QUEUE_TABLE_INFO['columns'],
        db_config={'db_type': 'sqlite', 'db_name': str(tmp_path / 'queue.db'),
                   'user': None, 'password': None, 'host': None, 'port': None},
    )
    try:
        now = datetime.now()
        queue.enqueue(task_id='task', task_type='DOC_ADD', task_status='SUCCESS', finished_at=now)
        # Simulate timestamps stored before and after switching database drivers.
        for stored in [now.isoformat(' '), now.replace(tzinfo=timezone(timedelta(hours=8))).isoformat()]:
            with queue._sql_manager.engine.begin() as conn:
                conn.execute(sqlalchemy.text('UPDATE finished_time_test SET finished_at=:value'), {'value': stored})
            assert queue.peek()['finished_at'] == stored
        queue.clear()
        assert queue.peek() is None
    finally:
        queue._sql_manager.engine.dispose()
