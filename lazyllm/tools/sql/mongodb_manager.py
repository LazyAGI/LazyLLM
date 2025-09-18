import json
from contextlib import contextmanager
from urllib.parse import quote_plus
import pydantic

from lazyllm.thirdparty import pymongo

from .db_manager import DBManager, DBResult, DBStatus


class CollectionDesc(pydantic.BaseModel):
    summary: str = ''
    schema_type: dict
    schema_desc: dict


class MongoDBManager(DBManager):
    """MongoDBManager is a specialized tool for interacting with MongoB databases.
It provides methods to check the connection, obtain the database connection object, and execute query.

Args:
   user (str): MongoDB username
    password (str): MongoDB password
    host (str): MongoDB server address
    port (int): MongoDB server port
    db_name (str): Database name
    collection_name (str): Collection name
    **kwargs: Additional configuration parameters including:
        - options_str (str): Connection options string
        - collection_desc_dict (dict): Collection description dictionary


Examples:
    >>> from lazyllm.components import MongoDBManager
    >>> mgr = MongoDBManager(
    ...     user="admin",
    ...     password="123456",
    ...     host="localhost",
    ...     port=27017,
    ...     db_name="mydb",
    ...     collection_name="books"
    ... )
    >>> result = mgr.execute_query('[{"$match": {"author": "Tolstoy"}}]')
    >>> print(result)
    ... '[{"title": "War and Peace", "author": "Tolstoy"}]'
    """
    MAX_TIMEOUT_MS = 5000

    def __init__(self, user: str, password: str, host: str, port: int, db_name: str, collection_name: str, **kwargs):
        super().__init__(db_type='mongodb')
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._db_name = db_name
        self._collection_name = collection_name
        self._collection = None
        self._options_str = kwargs.get('options_str')
        self._conn_url = self._gen_conn_url()
        self._collection_desc_dict = kwargs.get('collection_desc_dict')

    @property
    def db_name(self):
        return self._db_name

    @property
    def collection_name(self):
        return self._collection_name

    def _gen_conn_url(self) -> str:
        password = quote_plus(self._password)
        conn_url = (f'{self._db_type}://{self._user}:{password}@{self._host}:{self._port}/'
                    f'{("?" + self._options_str) if self._options_str else ""}')
        return conn_url

    @contextmanager
    def get_client(self):
        """This is a context manager that creates a database session, yields it for use, and closes the session when done.
Usage example:

with mongodb_manager.get_client() as client:
    all_dbs = client.list_database_names()

**Returns:**

- pymongo.MongoClient: MongoDB client used to connect to MongoDB database
"""
        client = pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS)
        try:
            yield client
        finally:
            client.close()

    @property
    def desc(self):
        if self._desc is None:
            self.set_desc(schema_desc_dict=self._collection_desc_dict)
        return self._desc

    def set_desc(self, schema_desc_dict: dict):
        """When using MongoDBManager with LLM to query documents in natural language, set descriptions for the necessary keywords. Note that all relevant keywords needed for queries must be provided because MongoDB cannot obtain like structural information like a SQL database.

Args:
    tables_desc_dict (dict): descriptive comment for documents
"""
        self._collection_desc_dict = schema_desc_dict
        if schema_desc_dict is None:
            with self.get_client() as client:
                egs_one = client[self._db_name][self._collection_name].find_one()
                if egs_one is not None:
                    self._desc = 'Collection Example:\n'
                    self._desc += json.dumps(egs_one, ensure_ascii=False, indent=4)
        else:
            self._desc = ''
            try:
                collection_desc = CollectionDesc.model_validate(schema_desc_dict)
            except pydantic.ValidationError as e:
                raise ValueError(f'Validate input schema_desc_dict failed: {str(e)}')
            if not self._is_dict_all_str(collection_desc.schema_type):
                raise ValueError('schema_type shouble be str or nested str dict')
            if not self._is_dict_all_str(collection_desc.schema_desc):
                raise ValueError('schema_desc shouble be str or nested str dict')
            if collection_desc.summary:
                self._desc += f'Collection summary: {collection_desc.summary}\n'
            self._desc += 'Collection schema:\n'
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)
            self._desc += 'Collection schema description:\n'
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)

    def check_connection(self) -> DBResult:
        """Check the current connection status of the MongoDBManager.

**Returns:**

- DBResult: DBResult.status True if the connection is successful, False if it fails. DBResult.detail contains failure information.
"""
        try:
            with pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS) as client:
                _ = client.server_info()
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def execute_query(self, statement) -> str:
        str_result = ''
        try:
            pipeline_list = json.loads(statement)
            with self.get_client() as client:
                collection = client[self._db_name][self._collection_name]
                result = list(collection.aggregate(pipeline_list))
                str_result = json.dumps(result, ensure_ascii=False, default=self._serialize_uncommon_type)
        except Exception as e:
            str_result = f'MongoDB ERROR: {str(e)}'
        return str_result
