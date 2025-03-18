import unittest
from lazyllm.tools import SqlCall, SqlManager, DBStatus
import lazyllm
from .utils import SqlEgsData, get_db_init_keywords
import datetime
import re
import pytest


@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
class TestSqlManager(unittest.TestCase):
    @classmethod
    def clean_obsolete_tables(cls, sql_manager: SqlManager):
        today = datetime.datetime.now()
        pattern = r"^(?:employee|sales)_(\d{8})_(\w+)"
        OBSOLETE_DAYS = 2
        existing_tables = sql_manager.get_all_tables()
        for table_name in existing_tables:
            match = re.match(pattern, table_name)
            if not match:
                continue
            table_create_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            delta = (today - table_create_date).days
            if delta >= OBSOLETE_DAYS:
                sql_manager.drop_table(table_name)

    @classmethod
    def setUpClass(cls):
        cls.sql_managers: list[SqlManager] = [SqlManager("SQLite", None, None, None, None, db_name=":memory:",
                                                         tables_info_dict=SqlEgsData.TEST_TABLES_INFO)]
        # MySQL has been tested with online database.
        for db_type in ["PostgreSQL"]:
            username, password, host, port, database = get_db_init_keywords(db_type)
            cls.sql_managers.append(SqlManager(db_type, username, password, host, port, database,
                                               tables_info_dict=SqlEgsData.TEST_TABLES_INFO))
        for sql_manager in cls.sql_managers:
            cls.clean_obsolete_tables(sql_manager)

            for table_name in SqlEgsData.TEST_TABLES:
                sql_manager.execute_commit(f"DELETE FROM {table_name}")
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                sql_manager.execute_commit(insert_script)

        # Recommend to use sensenova, gpt-4o, qwen online model
        sql_llm = lazyllm.OnlineChatModule(source="qwen")
        cls.sql_calls: list[SqlCall] = []
        for sql_manager in cls.sql_managers:
            cls.sql_calls.append(SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=True))

    @classmethod
    def tearDownClass(cls):
        # restore to clean database
        for sql_manager in cls.sql_managers:
            for table_name in SqlEgsData.TEST_TABLES:
                db_result = sql_manager.drop_table(table_name)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_status(self):
        for sql_manager in self.sql_managers:
            db_result = sql_manager.check_connection()
            assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_orm_operation(self):
        for sql_manager in self.sql_managers:
            table_name = SqlEgsData.TEST_TABLES[0]
            TableCls = sql_manager.get_table_orm_class(table_name)
            sql_manager.insert_values(table_name, SqlEgsData.TEST_EMPLOYEE_INSERT_VALS)

            with sql_manager.get_session() as session:
                item = session.query(TableCls).filter(TableCls.employee_id == 1111).first()
                assert item.name == "四一"

    def test_manager_table_delete_insert_query(self):
        # 1. Delete, as rows already exists during setUp
        for sql_manager in self.sql_managers:
            for table_name in SqlEgsData.TEST_TABLES:
                sql_manager.execute_commit(f"DELETE FROM {table_name}")
            str_results = sql_manager.execute_query(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertNotIn("销售一部", str_results)

        # 2. Insert, restore rows
        for sql_manager in self.sql_managers:
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                sql_manager.execute_commit(insert_script)
            str_results = sql_manager.execute_query(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertIn("销售一部", f"Query: {SqlEgsData.TEST_QUERY_SCRIPTS}; result: {str_results}")

    def test_llm_query_online(self):
        for sql_call in self.sql_calls:
            str_results = sql_call("去年一整年销售额最多的员工是谁，销售额是多少？")
            self.assertIn("张三", str_results)

            str_results = sql_call("删除员工信息表")
            self.assertIn("DROP TABLE", str_results.upper())


if __name__ == "__main__":
    unittest.main()
