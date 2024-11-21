import unittest
from lazyllm.tools import SQLiteManger, SqlCall, SqlManager
import lazyllm
from .utils import SqlEgsData, get_sql_init_keywords
import datetime
import re


class TestSqlManager(unittest.TestCase):
    @classmethod
    def clean_obsolete_tables(cls, sql_manager):
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
                sql_manager._drop_table_by_name(table_name)

    @classmethod
    def setUpClass(cls):
        cls.sql_managers: list[SqlManager] = [SQLiteManger(":memory:", SqlEgsData.TEST_TABLES_INFO)]
        for db_type in ["PostgreSQL"]:
            username, password, host, port, database = get_sql_init_keywords(db_type)
            cls.sql_managers.append(
                SqlManager(db_type, username, password, host, port, database, SqlEgsData.TEST_TABLES_INFO)
            )
        for sql_manager in cls.sql_managers:
            cls.clean_obsolete_tables(sql_manager)
            for table_name in SqlEgsData.TEST_TABLES:
                rt, err_msg = sql_manager._delete_rows_by_name(table_name)
                assert rt, err_msg
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                sql_manager.execute_sql_update(insert_script)

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
                rt, err_msg = sql_manager._drop_table_by_name(table_name)
                assert rt, f"sql_manager table {table_name} error: {err_msg}"

    def test_manager_status(self):
        for sql_manager in self.sql_managers:
            rt, err_msg = sql_manager.check_connection()
            assert rt, err_msg
            assert sql_manager.err_code == 0

    def test_manager_table_create_drop(self):
        for sql_manager in self.sql_managers:
            # 1. drop tables
            for table_name in SqlEgsData.TEST_TABLES:
                rt, err_msg = sql_manager._drop_table_by_name(table_name)
                assert rt, err_msg
            existing_tables = set(sql_manager.get_all_tables())
            for table_name in SqlEgsData.TEST_TABLES:
                assert table_name not in existing_tables
            # 2. create table
            rt, err_msg = sql_manager.reset_tables(SqlEgsData.TEST_TABLES_INFO)
            assert rt, err_msg

            # 3. restore rows
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                rt, err_msg = sql_manager.execute_sql_update(insert_script)
                assert rt, err_msg

    def test_manager_table_delete_insert_query(self):
        # 1. Delete, as rows already exists during setUp
        for sql_manager in self.sql_managers:
            for table_name in SqlEgsData.TEST_TABLES:
                rt, err_msg = sql_manager._delete_rows_by_name(table_name)
                assert rt, err_msg
            str_results = sql_manager.get_query_result_in_json(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertNotIn("销售一部", str_results)

        # 2. Insert, restore rows
        for sql_manager in self.sql_managers:
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                rt, err_msg = sql_manager.execute_sql_update(insert_script)
                assert rt, err_msg
            str_results = sql_manager.get_query_result_in_json(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertIn("销售一部", str_results)

    def test_get_talbes(self):
        for sql_manager in self.sql_managers:
            tables_desc = sql_manager.get_tables_desc()
        self.assertIn("employee", tables_desc)
        self.assertIn("sales", tables_desc)

    def test_llm_query_online(self):
        for sql_call in self.sql_calls:
            str_results = sql_call("去年一整年销售额最多的员工是谁，销售额是多少？")
            self.assertIn("张三", str_results)


if __name__ == "__main__":
    unittest.main()
