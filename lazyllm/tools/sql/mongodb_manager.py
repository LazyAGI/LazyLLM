import json
from contextlib import contextmanager
from urllib.parse import quote_plus
import pydantic

from lazyllm.thirdparty import pymongo

from .db_manager import DBManager, DBResult, DBStatus


class CollectionDesc(pydantic.BaseModel):
    summary: str = ""
    schema_type: dict
    schema_desc: dict


class MongoDBManager(DBManager):
    MAX_TIMEOUT_MS = 5000

    def __init__(self, user: str, password: str, host: str, port: int, db_name: str, collection_name: str,
                 options_str="", collection_desc_dict: dict = None):
        super().__init__(db_type="mongodb")
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.options_str = options_str
        self._collection = None
        self.collection_desc_dict = None
        self._conn_url = self._gen_conn_url()

    def _gen_conn_url(self) -> str:
        password = quote_plus(self.password)
        conn_url = (f"{self._db_type}://{self.user}:{password}@{self.host}:{self.port}/"
                    f"{('?' + self.options_str) if self.options_str else ''}")
        return conn_url

    @contextmanager
    def get_client(self):
        """
        Get the client object with the context manager.
        Use client to manage database.

        Usage:

        >>> with mongodb_manager.get_client() as client:
        >>>     all_dbs = client.list_database_names()
        """
        client = pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS)
        try:
            yield client
        finally:
            client.close()

    @property
    def desc(self):
        if self._desc is None:
            self.set_desc(schema_desc_dict=self.collection_desc_dict)
        return self._desc

    def set_desc(self, schema_desc_dict: dict):
        self.collection_desc_dict = schema_desc_dict
        if schema_desc_dict is None:
            with self.get_client() as client:
                egs_one = client[self.db_name][self.collection_name].find_one()
                if egs_one is not None:
                    self._desc = "Collection Example:\n"
                    self._desc += json.dumps(egs_one, ensure_ascii=False, indent=4)
        else:
            self._desc = ""
            try:
                collection_desc = CollectionDesc.model_validate(schema_desc_dict)
            except pydantic.ValidationError as e:
                raise ValueError(f"Validate input schema_desc_dict failed: {str(e)}")
            if not self._is_dict_all_str(collection_desc.schema_type):
                raise ValueError("schema_type shouble be str or nested str dict")
            if not self._is_dict_all_str(collection_desc.schema_desc):
                raise ValueError("schema_desc shouble be str or nested str dict")
            if collection_desc.summary:
                self._desc += f"Collection summary: {collection_desc.summary}\n"
            self._desc += "Collection schema:\n"
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)
            self._desc += "Collection schema description:\n"
            self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)

    def check_connection(self) -> DBResult:
        try:
            with pymongo.MongoClient(self._conn_url, serverSelectionTimeoutMS=self.MAX_TIMEOUT_MS) as client:
                _ = client.server_info()
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def execute_query(self, statement) -> str:
        str_result = ""
        try:
            pipeline_list = json.loads(statement)
            with self.get_client() as client:
                collection = client[self.db_name][self.collection_name]
                result = list(collection.aggregate(pipeline_list))
                str_result = json.dumps(result, ensure_ascii=False, default=self._serialize_uncommon_type)
        except Exception as e:
            str_result = f"MongoDB ERROR: {str(e)}"
        return str_result
