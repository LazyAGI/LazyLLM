import os
import re
import uuid
import datetime


UUID_HEX = str(uuid.uuid4().hex)
CURRENT_DAY = datetime.datetime.now().strftime("%Y%m%d")


class SqlEgsData:
    TEST_TABLES = [f"employee_{CURRENT_DAY}_{UUID_HEX}", f"sales_{CURRENT_DAY}_{UUID_HEX}"]
    TEST_TABLES_INFO = {
        "tables": [
            {
                "name": f"{TEST_TABLES[0]}",
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
                "name": f"{TEST_TABLES[1]}",
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
        f"INSERT INTO {TEST_TABLES[0]} VALUES (1, '张三', '销售一部');",
        f"INSERT INTO {TEST_TABLES[0]} VALUES (2, '李四', '销售二部');",
        f"INSERT INTO {TEST_TABLES[0]} VALUES (11, '王五', '销售三部');",
        f"INSERT INTO {TEST_TABLES[1]} VALUES (1, 8715.55, 8465.65, 24747.82, 3514.36);",
        f"INSERT INTO {TEST_TABLES[1]} VALUES (2, 4989.23, 5103.22, 4897.98, 5322.05);",
        f"INSERT INTO {TEST_TABLES[1]} VALUES (11, 5989.23, 6103.22, 2897.98, 3322.05);",
    ]
    TEST_QUERY_SCRIPTS = f"SELECT department from {TEST_TABLES[0]} WHERE employee_id=1;"


def get_sql_init_keywords(db_type):
    env_key = f"LAZYLLM_{db_type.replace(' ', '_')}_URL"
    conn_url = os.environ.get(env_key, None)
    assert conn_url is not None
    pattern = r"postgresql://(?P<username>[^:]+):(?P<password>.+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)"
    match = re.search(pattern, conn_url)
    assert match
    username = match.group("username")
    password = match.group("password")
    host = match.group("host")
    port = match.group("port")
    database = match.group("database")
    return username, password, host, port, database
