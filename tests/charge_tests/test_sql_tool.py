import unittest
from lazyllm.tools import SQLiteManger, SqlCall, SqlManager, DBStatus
import lazyllm
from pathlib import Path
from .utils import SqlEgsData, get_db_init_keywords
import datetime
import re


class TestSqlManager(unittest.TestCase):
    @classmethod
    def clean_obsolete_tables(cls, sql_manager: SqlManager):
        today = datetime.datetime.now()
        pattern = r"^(?:employee|sales)_(\d{8})_(\w+)"
        OBSOLETE_DAYS = 2
        db_result = sql_manager.get_all_tables()
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        existing_tables = db_result.result
        for table_name in existing_tables:
            match = re.match(pattern, table_name)
            if not match:
                continue
            table_create_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            delta = (today - table_create_date).days
            if delta >= OBSOLETE_DAYS:
                sql_manager.drop(table_name)

    @classmethod
    def setUpClass(cls):
        cls.sql_managers: list[SqlManager] = [SQLiteManger(":memory:", SqlEgsData.TEST_TABLES_INFO)]
        for db_type in ["PostgreSQL"]:
            username, password, host, port, database = get_db_init_keywords(db_type)
            cls.sql_managers.append(
                SqlManager(db_type, username, password, host, port, database, SqlEgsData.TEST_TABLES_INFO)
            )
        for sql_manager in cls.sql_managers:
            cls.clean_obsolete_tables(sql_manager)
            for table_name in SqlEgsData.TEST_TABLES:
                db_result = sql_manager.delete(table_name)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                db_result = sql_manager.execute(insert_script)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail

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
                db_result = sql_manager.drop(table_name)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_status(self):
        for sql_manager in self.sql_managers:
            db_result = sql_manager.check_connection()
            assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_table_create_drop(self):
        for sql_manager in self.sql_managers:
            # 1. drop tables
            for table_name in SqlEgsData.TEST_TABLES:
                db_result = sql_manager.drop(table_name)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail
            db_result = sql_manager.get_all_tables()
            assert db_result.status == DBStatus.SUCCESS, db_result.detail
            existing_tables = set(db_result.result)
            for table_name in SqlEgsData.TEST_TABLES:
                assert table_name not in existing_tables
            # 2. create table
            db_result = sql_manager.reset_table_info_dict(SqlEgsData.TEST_TABLES_INFO)
            assert db_result.status == DBStatus.SUCCESS, db_result.detail

            # 3. restore rows
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                db_result = sql_manager.execute(insert_script)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_table_delete_insert_query(self):
        # 1. Delete, as rows already exists during setUp
        for sql_manager in self.sql_managers:
            for table_name in SqlEgsData.TEST_TABLES:
                db_result = sql_manager.delete(table_name)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail
            str_results = sql_manager.execute_to_json(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertNotIn("销售一部", str_results)

        # 2. Insert, restore rows
        for sql_manager in self.sql_managers:
            for insert_script in SqlEgsData.TEST_INSERT_SCRIPTS:
                db_result = sql_manager.execute(insert_script)
                assert db_result.status == DBStatus.SUCCESS, db_result.detail
            str_results = sql_manager.execute_to_json(SqlEgsData.TEST_QUERY_SCRIPTS)
            self.assertIn("销售一部", f"Query: {SqlEgsData.TEST_QUERY_SCRIPTS}; result: {str_results}")

    def test_get_tables(self):
        for sql_manager in self.sql_managers:
            db_result = sql_manager.get_all_tables()
            assert db_result.status == DBStatus.SUCCESS, db_result.detail
            for table_name in SqlEgsData.TEST_TABLES:
                self.assertIn(table_name, db_result.result)

    def test_llm_query_online(self):
        for sql_call in self.sql_calls:
            str_results = sql_call("去年一整年销售额最多的员工是谁，销售额是多少？")
            self.assertIn("张三", str_results)


if __name__ == "__main__":
    unittest.main()
