import unittest
from lazyllm.tools import SQLiteManger, SqlCall, SqlManager
import lazyllm
import tempfile
from pathlib import Path
import uuid
from lazyllm import LightEngine
import os
import re
import json


class TestSQLite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filepath = str(Path(tempfile.gettempdir()) / f"{str(uuid.uuid4().hex)}.db")
        cls.db_filepath = filepath
        with open(filepath, "w") as _:
            pass
        tables_info = {
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
                    "comment": "销售额记录表",
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
        sql_tool = SQLiteManger(filepath, tables_info)
        assert not sql_tool.err_msg
        sql_tool.execute_sql_update("INSERT INTO employee VALUES (1, '张三', '销售一部');")
        sql_tool.execute_sql_update("INSERT INTO employee VALUES (2, '李四', '销售二部');")
        sql_tool.execute_sql_update("INSERT INTO sales VALUES (1, 8715.55, 8465.65, 24747.82, 3514.36);")
        sql_tool.execute_sql_update("INSERT INTO sales VALUES (2, 4989.23, 5103.22, 4897.98, 5322.05);")
        cls.sql_tool: SQLiteManger = sql_tool
        # Recommend to use sensenova, gpt-4o, qwen online model
        sql_llm = lazyllm.OnlineChatModule(source="sensenova")
        cls.sql_module: SqlCall = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)

    @classmethod
    def tearDownClass(cls):
        db_path = Path(cls.db_filepath)
        if db_path.is_file():
            db_path.unlink()

    def test_get_talbes(self):
        str_result = self.sql_tool.get_tables_desc()
        self.assertIn("employee", str_result)

    def test_sql_query(self):
        str_results = self.sql_tool.get_query_result_in_json("SELECT department from employee WHERE employee_id=1;")
        self.assertIn("销售一部", str_results)

    @unittest.skip("temporary skip")
    def test_llm_query(self):
        # 3. llm chat
        str_results = self.sql_module("去年一整年销售额最多的员工是谁，销售额是多少？")
        print(str_results)
        self.assertIn("张三", str_results)


class TestOnlineSql(unittest.TestCase):
    tables_info = {
        "tables": [
            {
                "name": "employee",
                "comment": "员工信息表",
                "columns": [
                    {
                        "name": "employee_id",
                        "data_type": "Integer",
                        "comment": "工号",
                        "nullable": True,
                        "is_primary_key": True,
                    },
                    {"name": "first_name", "data_type": "Text", "comment": "姓", "nullable": True},
                    {"name": "last_name", "data_type": "Text", "comment": "名", "nullable": True},
                    {"name": "department", "data_type": "Text", "comment": "部门", "nullable": True},
                ],
            }
        ]
    }

    def _get_sql_init_keywords(self, db_type):
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

    def test_sql_api(self):
        db_types = ["PostgreSQL"]
        sql_llm = lazyllm.OnlineChatModule(source="sensenova")
        for db_type in db_types:
            username, password, host, port, database = self._get_sql_init_keywords(db_type)
            sql_manager = SqlManager(db_type, username, password, host, port, database, self.tables_info)
            result_str = sql_manager.get_query_result_in_json("select * from employee where employee_id=3")
            json_obj = json.loads(result_str)
            assert len(json_obj) == 1
            sql_module: SqlCall = SqlCall(sql_llm, sql_manager, use_llm_for_sql_result=True)
            str_result = sql_module("员工编号是3的人来自哪个部门？")
            print("str_result:", str_result)

    @unittest.skip("temporary skip")
    def test_engine(self):
        db_types = ["PostgreSQL"]
        for db_type in db_types:
            username, password, host, port, database = self._get_sql_init_keywords(db_type)
            resources = [
                dict(
                    id="0",
                    kind="SqlManager",
                    name="sql_tool",
                    args=dict(
                        db_type=db_type,
                        user=username,
                        password=password,
                        host=host,
                        port=port,
                        db_name=database,
                        tabels_info_dict=self.tables_info,
                    ),
                ),
                dict(id="1", kind="OnlineLLM", name="llm", args=dict(source="sensenova")),
            ]
            nodes = [
                dict(
                    id="2",
                    kind="SqlCall",
                    name="sql_call",
                    args=dict(sql_tool="0", llm="1", tables=[], tables_desc="", sql_examples=""),
                )
            ]
            edges = [dict(iid="__start__", oid="2"), dict(iid="2", oid="__end__")]
            engine = LightEngine()
            engine.start(nodes, edges, resources)
            str_answer = engine.run("员工编号是3的人来自哪个部门？")
            print(str_answer)
            assert "销售三部" in str_answer


if __name__ == "__main__":
    unittest.main()
