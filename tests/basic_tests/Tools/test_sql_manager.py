from unittest.mock import MagicMock, patch

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
