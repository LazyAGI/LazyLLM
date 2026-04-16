import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy
from sqlalchemy.exc import OperationalError

from lazyllm.tools.sql.sql_manager import SqlManager


def _make_unknown_database_error():
    class _UnknownDatabaseError(Exception):
        args = (1049, "Unknown database 'lazyllm_doc_task'")

    return OperationalError('SELECT 1', {}, _UnknownDatabaseError())


def _make_conn_cm(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


class TestSqlManager(object):

    @patch('lazyllm.tools.sql.sql_manager.sqlalchemy.create_engine')
    def test_tidb_create_database_when_missing(self, mock_create_engine):
        probe_engine = MagicMock()
        probe_engine.connect.side_effect = _make_unknown_database_error()

        admin_conn = MagicMock()
        admin_engine = MagicMock()
        admin_engine.connect.return_value = _make_conn_cm(admin_conn)

        final_engine = MagicMock()
        mock_create_engine.side_effect = [probe_engine, admin_engine, final_engine]

        sql_manager = SqlManager('tidb', 'root', 'pwd', '127.0.0.1', 4000, 'lazyllm_doc_task')

        assert sql_manager.engine is final_engine
        assert mock_create_engine.call_args_list[1].args[0] == 'mysql+pymysql://root:pwd@127.0.0.1:4000/'
        assert str(admin_conn.execute.call_args.args[0]) == 'CREATE DATABASE IF NOT EXISTS `lazyllm_doc_task`'

    @patch('lazyllm.tools.sql.sql_manager.sqlalchemy.create_engine')
    def test_tidb_skip_create_database_when_exists(self, mock_create_engine):
        probe_engine = MagicMock()
        probe_engine.connect.return_value = _make_conn_cm(MagicMock())

        final_engine = MagicMock()
        mock_create_engine.side_effect = [probe_engine, final_engine]

        sql_manager = SqlManager('tidb', 'root', 'pwd', '127.0.0.1', 4000, 'lazyllm_doc_task')

        assert sql_manager.engine is final_engine
        assert mock_create_engine.call_count == 2

    @patch('lazyllm.tools.sql.sql_manager.sqlalchemy.create_engine')
    def test_tidb_primary_key_string_uses_varchar(self, mock_create_engine):
        sql_manager = SqlManager('tidb', 'root', 'pwd', '127.0.0.1', 4000, 'lazyllm_doc_task')

        primary_key_type = sql_manager._sql_type_for('string', is_primary_key=True)
        normal_string_type = sql_manager._sql_type_for('string')

        mock_create_engine.assert_not_called()
        assert isinstance(primary_key_type, sqlalchemy.String)
        assert primary_key_type.length == 255
        assert normal_string_type is sqlalchemy.Text


# ======================================================================
# Integration tests for get_session(session=...) and paginate(query, ...)
# backed by a real sqlite file so session/commit/rollback semantics are
# exercised end-to-end rather than mocked.
# ======================================================================


@pytest.fixture
def sqlite_sql_manager():
    tmp = tempfile.TemporaryDirectory(prefix='lazyllm_sqlmgr_test_')
    try:
        db_path = os.path.join(tmp.name, 'test.db')
        tables_info = {
            'tables': [
                {
                    'name': 'session_items',
                    'comment': 'Scratch table for session tests',
                    'columns': [
                        {'name': 'id', 'data_type': 'integer',
                         'is_primary_key': True, 'nullable': False},
                        {'name': 'value', 'data_type': 'string', 'nullable': False},
                    ],
                }
            ]
        }
        mgr = SqlManager(
            db_type='sqlite', user=None, password=None, host=None, port=None,
            db_name=db_path, tables_info_dict=tables_info,
        )
        try:
            yield mgr
        finally:
            mgr.dispose()
    finally:
        tmp.cleanup()


class TestGetSessionOptional:
    '''``SqlManager.get_session(session=None)`` — verify the helper-driven
    pattern where internal methods accept an optional session.'''

    def test_opens_new_session_when_none_passed(self, sqlite_sql_manager):
        Item = sqlite_sql_manager.get_table_orm_class('session_items')
        with sqlite_sql_manager.get_session() as session:
            session.add(Item(id=1, value='a'))
        # Implicit commit on context exit => row is visible to a fresh session.
        with sqlite_sql_manager.get_session() as session:
            assert session.query(Item).filter(Item.id == 1).first().value == 'a'

    def test_reuses_external_session_without_commit(self, sqlite_sql_manager):
        '''When a session is passed through, the inner context must neither
        commit nor close it — the outer owner still controls the transaction.'''
        Item = sqlite_sql_manager.get_table_orm_class('session_items')

        def _helper(mgr, value, session=None):
            # Typical pattern used by DocManager helpers.
            with mgr.get_session(session) as sess:
                sess.add(Item(id=2, value=value))

        with sqlite_sql_manager.get_session() as outer:
            _helper(sqlite_sql_manager, 'b', session=outer)
            # Row must exist within the caller's transaction.
            assert outer.query(Item).filter(Item.id == 2).first() is not None

        # Row persists after the outer commit.
        with sqlite_sql_manager.get_session() as session:
            assert session.query(Item).filter(Item.id == 2).first().value == 'b'

    def test_external_session_rollback_on_exception(self, sqlite_sql_manager):
        '''When a helper participates in an outer transaction, an exception
        inside the helper must propagate to the outer ``get_session`` and
        trigger its rollback — both the helper's write AND any prior write
        inside the outer block must disappear.'''
        Item = sqlite_sql_manager.get_table_orm_class('session_items')

        def _helper(mgr, session=None):
            with mgr.get_session(session) as sess:
                sess.add(Item(id=4, value='helper-write'))
                raise RuntimeError('simulated failure inside helper')

        with pytest.raises(RuntimeError, match='simulated failure'):
            with sqlite_sql_manager.get_session() as outer:
                outer.add(Item(id=3, value='outer-write'))
                _helper(sqlite_sql_manager, session=outer)

        with sqlite_sql_manager.get_session() as session:
            assert session.query(Item).filter(Item.id.in_([3, 4])).count() == 0, \
                'outer rollback must discard both outer-write and helper-write'

    def test_standalone_session_rollback_on_exception(self, sqlite_sql_manager):
        '''Without an external session, an exception inside the context must
        roll back the row the context manager itself opened.'''
        Item = sqlite_sql_manager.get_table_orm_class('session_items')

        with pytest.raises(RuntimeError):
            with sqlite_sql_manager.get_session() as session:
                session.add(Item(id=5, value='will-be-rolled-back'))
                raise RuntimeError('standalone failure')

        with sqlite_sql_manager.get_session() as session:
            assert session.query(Item).filter(Item.id == 5).first() is None

    def test_external_session_rollback_after_flush(self, sqlite_sql_manager):
        '''Production helpers (e.g. ``_upsert_doc``) call ``sess.flush()``
        before a failure point. The outer ``get_session`` must still roll back
        flushed rows — it's the transaction boundary, not the flush, that
        controls durability.
        '''
        Item = sqlite_sql_manager.get_table_orm_class('session_items')

        def _helper(mgr, session=None):
            with mgr.get_session(session) as sess:
                sess.add(Item(id=6, value='flushed'))
                sess.flush()  # row is sent to the DB but transaction is open
                raise RuntimeError('after-flush failure')

        with pytest.raises(RuntimeError, match='after-flush'):
            with sqlite_sql_manager.get_session() as outer:
                _helper(sqlite_sql_manager, session=outer)

        with sqlite_sql_manager.get_session() as session:
            assert session.query(Item).filter(Item.id == 6).first() is None, \
                'flushed row must be rolled back when the outer session unwinds'


class TestPaginate:
    '''``SqlManager.paginate(query, *, page, page_size)`` — domain-agnostic
    page envelope used by every listing endpoint.'''

    def _seed(self, mgr, n):
        Item = mgr.get_table_orm_class('session_items')
        with mgr.get_session() as session:
            for i in range(n):
                session.add(Item(id=i + 1, value=f'v{i:03d}'))

    def test_envelope_shape_and_total_matches(self, sqlite_sql_manager):
        self._seed(sqlite_sql_manager, 40)
        Item = sqlite_sql_manager.get_table_orm_class('session_items')
        with sqlite_sql_manager.get_session() as session:
            query = session.query(Item).order_by(Item.id.asc())
            result = SqlManager.paginate(query, page=2, page_size=10)
            assert set(result.keys()) == {'items', 'total', 'page', 'page_size'}
            assert result['total'] == 40
            assert result['page'] == 2
            assert result['page_size'] == 10
            assert [it.id for it in result['items']] == list(range(11, 21))

    def test_clamps_non_positive_page_and_page_size(self, sqlite_sql_manager):
        self._seed(sqlite_sql_manager, 3)
        Item = sqlite_sql_manager.get_table_orm_class('session_items')
        with sqlite_sql_manager.get_session() as session:
            query = session.query(Item).order_by(Item.id.asc())
            clamped = SqlManager.paginate(query, page=0, page_size=0)
            assert clamped['page'] == 1
            assert clamped['page_size'] == 1
            assert len(clamped['items']) == 1

            neg = SqlManager.paginate(query, page=-5, page_size=-10)
            assert neg['page'] == 1
            assert neg['page_size'] == 1

    def test_empty_result_returns_well_formed_page(self, sqlite_sql_manager):
        Item = sqlite_sql_manager.get_table_orm_class('session_items')
        with sqlite_sql_manager.get_session() as session:
            query = session.query(Item).filter(Item.id < 0)
            result = SqlManager.paginate(query, page=1, page_size=20)
            assert result == {'items': [], 'total': 0, 'page': 1, 'page_size': 20}

    def test_pages_across_full_dataset_sum_to_total(self, sqlite_sql_manager):
        self._seed(sqlite_sql_manager, 57)
        Item = sqlite_sql_manager.get_table_orm_class('session_items')
        with sqlite_sql_manager.get_session() as session:
            query = session.query(Item).order_by(Item.id.asc())
            all_ids = []
            totals = set()
            for page in range(1, 10):
                result = SqlManager.paginate(query, page=page, page_size=8)
                totals.add(result['total'])
                all_ids.extend(it.id for it in result['items'])
                if not result['items']:
                    break
            assert totals == {57}
            assert all_ids == list(range(1, 58))
