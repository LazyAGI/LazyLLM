import unittest
from lazyllm.tools import SQLiteManger, SqlCall, SqlManager
import lazyllm
import tempfile
from pathlib import Path
import uuid
import os
import re


def get_sql_init_keywords(db_type):
    env_key = f"LAZYLLM_{db_type.replace(' ', '_')}_URL"
    conn_url = os.environ.get(env_key, None)
    assert conn_url is not None
    pattern = r"postgresql://(?P<username>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)"
    match = re.search(pattern, conn_url)
    assert match
    username = match.group("username")
    password = match.group("password")
    host = match.group("host")
    port = match.group("port")
    database = match.group("database")
    return username, password, host, port, database


class TestSqlManager(unittest.TestCase):
    TEST_TABLES_INFO = {
        "tables": [
            {
                "name": "employee",
                "comment": "员工信息表",
                "columns": [
                    {
                        "name": "employee_id",
                        "data_type": "Integer",
                        "comment": "工号",
                        "nullable": False,
                        "is_primary_key": True,
                    },
                    {"name": "name", "data_type": "String", "comment": "姓名", "nullable": False},
                    {"name": "department", "data_type": "String", "comment": "部门", "nullable": False},
                ],
            },
            {
                "name": "sales",
                "comment": "销售额信息表",
                "columns": [
                    {
                        "name": "employee_id",
                        "data_type": "Integer",
                        "comment": "工号",
                        "nullable": False,
                        "is_primary_key": True,
                    },
                    {"name": "q1_2023", "data_type": "Float", "comment": "2023年第1季度销售额", "nullable": False},
                    {"name": "q2_2023", "data_type": "Float", "comment": "2023年第2季度销售额", "nullable": False},
                    {"name": "q3_2023", "data_type": "Float", "comment": "2023年第3季度销售额", "nullable": False},
                    {"name": "q4_2023", "data_type": "Float", "comment": "2023年第4季度销售额", "nullable": False},
                ],
            },
        ]
    }
    TEST_INSERT_SCRIPTS = [
        "INSERT INTO employee VALUES (1, '张三', '销售一部');",
        "INSERT INTO employee VALUES (2, '李四', '销售二部');",
        "INSERT INTO employee VALUES (3, '王五', '销售三部');",
        "INSERT INTO sales VALUES (1, 8715.55, 8465.65, 24747.82, 3514.36);",
        "INSERT INTO sales VALUES (2, 4989.23, 5103.22, 4897.98, 5322.05);",
        "INSERT INTO sales VALUES (3, 5989.23, 6103.22, 2897.98, 3322.05);",
    ]
    TEST_TABLES = ["employee", "sales"]

    @classmethod
    def setUpClass(cls):
        cls.sql_managers: list[SqlManager] = []

        filepath = str(Path(tempfile.gettempdir()) / f"{str(uuid.uuid4().hex)}.db")
        cls.db_filepath = filepath
        with open(filepath, "w") as _:
            pass
        cls.sql_managers.append(SQLiteManger(filepath, cls.TEST_TABLES_INFO))
        for db_type in ["PostgreSQL"]:
            username, password, host, port, database = get_sql_init_keywords(db_type)
            cls.sql_managers.append(SqlManager(db_type, username, password, host, port, database, cls.TEST_TABLES_INFO))
        for sql_manager in cls.sql_managers:
            for table_name in cls.TEST_TABLES:
                rt, err_msg = sql_manager._delete_rows_by_name(table_name)
                assert rt, err_msg
            for insert_script in cls.TEST_INSERT_SCRIPTS:
                sql_manager.execute_sql_update(insert_script)

        # Recommend to use sensenova, gpt-4o, qwen online model
        sql_llm = lazyllm.OnlineChatModule(source="sensenova")
        cls.sql_calls: list[SqlCall] = []
        for sql_manager in cls.sql_managers:
            cls.sql_calls.append(SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=True))

    @classmethod
    def tearDownClass(cls):
        # restore to clean database
        for sql_manager in cls.sql_managers:
            for table_name in cls.TEST_TABLES:
                rt, err_msg = sql_manager._drop_table_by_name(table_name)
                assert rt, f"sql_manager table {table_name} error: {err_msg}"
        db_path = Path(cls.db_filepath)
        if db_path.is_file():
            db_path.unlink()

    def test_manager_status(self):
        for sql_manager in self.sql_managers:
            rt, err_msg = sql_manager.check_connection()
            assert rt, err_msg
            assert sql_manager.err_code == 0

    def test_manager_table_create_drop(self):
        for sql_manager in self.sql_managers:
            # 1. drop tables
            for table_name in self.TEST_TABLES:
                rt, err_msg = sql_manager._drop_table_by_name(table_name)
                assert rt, err_msg
            existing_tables = sql_manager.get_all_tables()
            assert len(existing_tables) == 0
            # 2. create table
            rt, err_msg = sql_manager.reset_tables(self.TEST_TABLES_INFO)
            assert rt, err_msg

            # 3. restore rows
            for insert_script in self.TEST_INSERT_SCRIPTS:
                rt, err_msg = sql_manager.execute_sql_update(insert_script)
                assert rt, err_msg

    def test_manager_table_delete_insert_query(self):
        query_script = "SELECT department from employee WHERE employee_id=1;"
        # 1. Delete, as rows already exists during setUp
        for sql_manager in self.sql_managers:
            for table_name in self.TEST_TABLES:
                rt, err_msg = sql_manager._delete_rows_by_name(table_name)
                assert rt, err_msg
            str_results = sql_manager.get_query_result_in_json(query_script)
            self.assertNotIn("销售一部", str_results)

        # 2. Insert, restore rows
        for sql_manager in self.sql_managers:
            for insert_script in self.TEST_INSERT_SCRIPTS:
                rt, err_msg = sql_manager.execute_sql_update(insert_script)
                assert rt, err_msg
            str_results = sql_manager.get_query_result_in_json(query_script)
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

    def test_llm_query_local(self):
        local_llm = lazyllm.TrainableModule("internlm2-chat-20b").deploy_method(lazyllm.deploy.vllm).start()
        sql_call = SqlCall(local_llm, self.sql_managers[0], use_llm_for_sql_result=True, return_trace=True)
        str_results = sql_call("员工编号是3的人来自哪个部门？")
        self.assertIn("销售三部", str_results)


if __name__ == "__main__":
    unittest.main()
