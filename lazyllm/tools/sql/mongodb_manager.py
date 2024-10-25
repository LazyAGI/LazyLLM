import json
import pydantic
from lazyllm.thirdparty import pymongo
from .db_manager import DBManager, DBStatus, DBResult


class CollectionDesc(pydantic.BaseModel):
    summary: str = ""
    schema_type: dict
    schema_desc: dict


class MongoDBManager(DBManager):
    def __init__(self, user, password, host, port, db_name, collection_name, options_str=""):
        result = self.reset_client(user, password, host, port, db_name, collection_name, options_str)
        self.status, self.detail = result.status, result.detail
        if self.status != DBStatus.SUCCESS:
            raise ValueError(self.detail)

    def reset_client(self, user, password, host, port, db_name, collection_name, options_str="") -> DBResult:
        self._db_type = "mongodb"
        self.status = DBStatus.SUCCESS
        self.detail = ""
        conn_url = f"{self._db_type}://{user}:{password}@{host}:{port}/"
        self._conn_url = conn_url
        self._db_name = db_name
        self._collection_name = collection_name
        if options_str:
            self._extra_fields = {
                key: value for key_value in options_str.split("&") for key, value in (key_value.split("="),)
            }
        else:
            self._extra_fields = {}
        self._client = pymongo.MongoClient(self._conn_url)
        result = self.check_connection()
        self._collection = self._client[self._db_name][self._collection_name]
        self._desc = {}
        if result.status != DBStatus.SUCCESS:
            return result
        """
        if db_name not in self.client.list_database_names():
            return DBResult(status=DBStatus.FAIL, detail=f"Database {db_name} not found")
        if collection_name not in self.client[db_name].list_collection_names():
            return DBResult(status=DBStatus.FAIL, detail=f"Collection {collection_name} not found")
        """
        return DBResult()

    def check_connection(self) -> DBResult:
        try:
            # check connection status
            _ = self._client.server_info()
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def get_all_collections(self):
        return DBResult(result=self._client[self._db_name].list_collection_names())

    def drop_database(self) -> DBResult:
        if self.status != DBStatus.SUCCESS:
            return DBResult(status=self.status, detail=self.detail, result=None)
        self._client.drop_database(self._db_name)
        return DBResult()

    def drop_collection(self, collection_name) -> DBResult:
        db = self._client[self._db_name]
        db[collection_name].drop()
        return DBResult()

    def insert(self, statement):
        if isinstance(statement, dict):
            self._collection.insert_one(statement)
        elif isinstance(statement, list):
            self._collection.insert_many(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail=f"statement type {type(statement)} not supported", result=None)
        return DBResult()

    def update(self, filter: dict, value: dict, is_many: bool = True):
        if is_many:
            self._collection.update_many(filter, value)
        else:
            self._collection.update_one(filter, value)
        return DBResult()

    def delete(self, filter: dict, is_many: bool = True):
        if is_many:
            self._collection.delete_many(filter)
        else:
            self._collection.delete_one(filter)

    def select(self, query, projection: dict[str, bool] = None, limit: int = None):
        if limit is None:
            result = self._collection.find(query, projection)
        else:
            result = self._collection.find(query, projection).limit(limit)
        return DBResult(result=list(result))

    def execute(self, statement):
        try:
            pipeline_list = json.loads(statement)
            result = self._collection.aggregate(pipeline_list)
            return DBResult(result=list(result))
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def execute_to_json(self, statement) -> str:
        dbresult = self.execute(statement)
        if dbresult.status != DBStatus.SUCCESS:
            self.status, self.detail = dbresult.status, dbresult.detail
            return ""
        str_result = json.dumps(dbresult.result, ensure_ascii=False, default=self._serialize_uncommon_type)
        return str_result

    @property
    def desc(self):
        return self._desc

    def set_desc(self, schema_and_desc: dict) -> DBResult:
        self._desc = ""
        try:
            collection_desc = CollectionDesc.model_validate(schema_and_desc)
        except pydantic.ValidationError as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))
        if not self._validate_desc(collection_desc.schema_type) or not self._validate_desc(collection_desc.schema_desc):
            err_msg = "key and value in desc shoule be str or nested str dict"
            return DBResult(status=DBStatus.FAIL, detail=err_msg)
        if collection_desc.summary:
            self._desc += f"Collection summary: {collection_desc.summary}\n"
        self._desc += "Collection schema:\n"
        self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)
        self._desc += "Collection schema description:\n"
        self._desc += json.dumps(collection_desc.schema_type, ensure_ascii=False, indent=4)
        return DBResult()
