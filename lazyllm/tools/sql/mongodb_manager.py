import json
import pydantic
from pymongo import MongoClient
from .db_manager import DBManager, DBStatus, DBResult


class CollectionDesc(pydantic.BaseModel):
    schema: dict
    schema_desc: dict


class MonogDBManager(DBManager):
    def __init__(self, user, password, host, port, db_name, collection_name, options_str=""):
        result = self.reset_client(user, password, host, port, db_name, collection_name, options_str)
        self.status, self.detail = result.status, result.detail
        if self.status != DBStatus.SUCCESS:
            raise ValueError(self.detail)

    def reset_client(self, user, password, host, port, db_name, collection_name, options_str="") -> DBResult:
        db_type_lower = "mongodb"
        self.status = DBStatus.SUCCESS
        self.detail = ""
        conn_url = f"{db_type_lower}://{user}:{password}@{host}:{port}/"
        self.conn_url = conn_url
        self.db_name = db_name
        self.collection_name = collection_name
        if options_str:
            extra_fields = {
                key: value for key_value in options_str.split("&") for key, value in (key_value.split("="),)
            }
        self.extra_fields = extra_fields
        self.client = MongoClient(self.conn_url)
        result = self.check_connection()
        self.collection = self.client[self.db_name][self.collection_name]
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
            _ = self.client.server_info()
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def drop_database(self) -> DBResult:
        if self.status != DBStatus.SUCCESS:
            return DBResult(status=self.status, detail=self.detail, result=None)
        self.client.drop_database(self.db_name)
        return DBResult()

    def drop_collection(self) -> DBResult:
        db = self.client[self.db_name]
        db[self.collection_name].drop()
        return DBResult()

    def insert(self, statement):
        if isinstance(statement, dict):
            self.collection.insert_one(statement)
        elif isinstance(statement, list):
            self.collection.insert_many(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail=f"statement type {type(statement)} not supported", result=None)
        return DBResult()

    def update(self, filter: dict, value: dict, is_many: bool = True):
        if is_many:
            self.collection.update_many(filter, value)
        else:
            self.collection.update_one(filter, value)
        return DBResult()

    def delete(self, filter: dict, is_many: bool = True):
        if is_many:
            self.collection.delete_many(filter)
        else:
            self.collection.delete_one(filter)

    def select(self, query, projection: dict[str, bool] = None, limit: int = None):
        if limit is not None:
            result = self.collection.find(query, projection)
        else:
            result = self.collection.find(query, projection).limit(limit)
        return DBResult(result=list(result))

    def execute(self, statement):
        try:
            pipeline_list = json.loads(statement)
            result = self.collection.aggregate(pipeline_list)
            return DBResult(result=list(result))
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    def execute_to_json(self, statement) -> str:
        dbresult = self.execute(statement)
        if dbresult.status != DBStatus:
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
        if not self._validate_desc(collection_desc.schema) or not self._validate_desc(collection_desc.schema_desc):
            err_msg = "key and value in desc shoule be str or nested str dict"
            return DBResult(status=DBStatus.FAIL, detail=err_msg)
        self.desc = "Collection schema:\n"
        self.desc += json.dumps(collection_desc.schema, ensure_ascii=False, indent=4)
        self.desc += "Collection schema description:\n"
        self.desc += json.dumps(collection_desc.schema, ensure_ascii=False, indent=4)
        return DBResult()
