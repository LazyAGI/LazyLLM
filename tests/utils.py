import os
import re
import uuid
import datetime


UUID_HEX = str(uuid.uuid4().hex)
CURRENT_DAY = datetime.datetime.now().strftime('%Y%m%d')


class SqlEgsData:
    TEST_TABLES = [f'employee_{CURRENT_DAY}_{UUID_HEX}', f'sales_{CURRENT_DAY}_{UUID_HEX}']
    TEST_TABLES_INFO = {
        'tables': [
            {
                'name': f'{TEST_TABLES[0]}',
                'comment': '员工信息表',
                'columns': [
                    {
                        'name': 'employee_id',
                        'data_type': 'Integer',
                        'comment': '工号',
                        'nullable': False,
                        'is_primary_key': True,
                    },
                    {'name': 'name', 'data_type': 'String', 'comment': '姓名', 'nullable': False},
                    {'name': 'department', 'data_type': 'String', 'comment': '部门', 'nullable': False},
                ],
            },
            {
                'name': f'{TEST_TABLES[1]}',
                'comment': '销售额信息表',
                'columns': [
                    {
                        'name': 'employee_id',
                        'data_type': 'Integer',
                        'comment': '工号',
                        'nullable': False,
                        'is_primary_key': True,
                    },
                    {'name': 'q1_2023', 'data_type': 'Float', 'comment': '2023年第1季度销售额', 'nullable': False},
                    {'name': 'q2_2023', 'data_type': 'Float', 'comment': '2023年第2季度销售额', 'nullable': False},
                    {'name': 'q3_2023', 'data_type': 'Float', 'comment': '2023年第3季度销售额', 'nullable': False},
                    {'name': 'q4_2023', 'data_type': 'Float', 'comment': '2023年第4季度销售额', 'nullable': False},
                ],
            },
        ]
    }

    TEST_INSERT_SCRIPTS = [
        f'INSERT INTO {TEST_TABLES[0]} VALUES (1, \'张三\', \'销售一部\');',
        f'INSERT INTO {TEST_TABLES[0]} VALUES (2, \'李四\', \'销售二部\');',
        f'INSERT INTO {TEST_TABLES[0]} VALUES (11, \'王五\', \'销售三部\');',
        f'INSERT INTO {TEST_TABLES[1]} VALUES (1, 8715.55, 8465.65, 24747.82, 3514.36);',
        f'INSERT INTO {TEST_TABLES[1]} VALUES (2, 4989.23, 5103.22, 4897.98, 5322.05);',
        f'INSERT INTO {TEST_TABLES[1]} VALUES (11, 5989.23, 6103.22, 2897.98, 3322.05);',
    ]
    TEST_EMPLOYEE_INSERT_VALS = [
        {'employee_id': 1111, 'name': '四一', 'department': 'IT'},
        {'employee_id': 11111, 'name': '五一', 'department': 'IT'}
    ]
    TEST_QUERY_SCRIPTS = f'SELECT department from {TEST_TABLES[0]} WHERE employee_id=1;'

class MongoDBEgsData:
    COLLECTION_NAME = f'america_{CURRENT_DAY}_{UUID_HEX}'
    COLLECTION_SCHEMA_TYPE = {
        '_id': 'string',
        'city': 'string',
        'state': 'string',
        'pop': 'int',
        'loc': {'type': 'string', 'coordinates': 'array of float'},
    }
    COLLECTION_SCHEMA_DESC = {
        'city': '城市名',
        'state': '两个字母的州名缩写',
        'pop': '人口数量',
        'loc': '城市的经纬度',
    }

    COLLECTION_DATA = [
        {
            'city': 'New York',
            'state': 'NY',
            'pop': 8419600,
            'loc': {'type': 'Point', 'coordinates': [-74.0060, 40.7128]},
        },
        {
            'city': 'Los Angeles',
            'state': 'CA',
            'pop': 3980400,
            'loc': {'type': 'Point', 'coordinates': [-118.2437, 34.0522]},
        },
        {
            'city': 'Chicago',
            'state': 'IL',
            'pop': 2716000,
            'loc': {'type': 'Point', 'coordinates': [-87.6298, 41.8781]},
        },
        {
            'city': 'Houston',
            'state': 'TX',
            'pop': 2328000,
            'loc': {'type': 'Point', 'coordinates': [-95.3698, 29.7604]},
        },
        {
            'city': 'Phoenix',
            'state': 'AZ',
            'pop': 1690000,
            'loc': {'type': 'Point', 'coordinates': [-112.0740, 33.4484]},
        },
        {
            'city': 'Philadelphia',
            'state': 'PA',
            'pop': 1584200,
            'loc': {'type': 'Point', 'coordinates': [-75.1652, 39.9526]},
        },
        {
            'city': 'San Antonio',
            'state': 'TX',
            'pop': 1547000,
            'loc': {'type': 'Point', 'coordinates': [-98.4936, 29.4241]},
        },
        {
            'city': 'San Diego',
            'state': 'CA',
            'pop': 1423800,
            'loc': {'type': 'Point', 'coordinates': [-117.1611, 32.7157]},
        },
        {'city': 'Dallas', 'state': 'TX', 'pop': 1343000, 'loc': {'type': 'Point', 'coordinates': [-96.7970, 32.7767]}},
        {
            'city': 'San Jose',
            'state': 'CA',
            'pop': 1028000,
            'loc': {'type': 'Point', 'coordinates': [-121.8863, 37.3382]},
        },
    ]


def get_db_init_keywords(db_type: str):
    env_key = f'LAZYLLM_{db_type.replace(" ", "_")}_URL'
    conn_url = os.environ.get(env_key, None)
    assert conn_url is not None
    pattern = (
        rf'{db_type.lower()}://(?P<username>[^:]+):(?P<password>.+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)'
    )
    match = re.search(pattern, conn_url)
    assert match
    username = match.group('username')
    password = match.group('password')
    host = match.group('host')
    port = match.group('port')
    database = match.group('database')
    return username, password, host, port, database

def get_path(supplier):
    return f'lazyllm/module/llms/onlinemodule/supplier/{supplier}.py'


def get_api_key(source):
    import lazyllm
    return lazyllm.config[f'{source}_api_key']
