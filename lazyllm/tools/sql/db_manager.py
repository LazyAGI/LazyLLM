from enum import Enum, unique
from typing import List, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod


@unique
class DBStatus(Enum):
    SUCCESS = 0
    FAIL = 1


class DBResult(BaseModel):
    status: DBStatus = DBStatus.SUCCESS
    detail: str = "Success"
    result: Union[List, None] = None


class DBManager(ABC):
    DB_TYPE_SUPPORTED = set(["postgresql", "mysql", "mssql", "sqlite", "mongodb"])
    DB_DRIVER_MAP = {"mysql": "pymysql"}

    def __init__(
        self,
        db_type: str,
        user: str,
        password: str,
        host: str,
        port: int,
        db_name: str,
        options_str: str = "",
    ) -> None:
        db_result = self.reset_engine(db_type, user, password, host, port, db_name, options_str)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(db_result.detail)

    def reset_engine(self, db_type, user, password, host, port, db_name, options_str):
        db_type_lower = db_type.lower()
        self.status = DBStatus.SUCCESS
        self.detail = ""
        self._db_type = db_type_lower
        if db_type_lower not in self.DB_TYPE_SUPPORTED:
            return DBResult(status=DBStatus.FAIL, detail=f"{db_type} not supported")
        if db_type_lower in self.DB_DRIVER_MAP:
            conn_url = (
                f"{db_type_lower}+{self.DB_DRIVER_MAP[db_type_lower]}://{user}:{password}@{host}:{port}/{db_name}"
            )
        else:
            conn_url = f"{db_type_lower}://{user}:{password}@{host}:{port}/{db_name}"
        self._conn_url = conn_url
        self._desc = ""

    @abstractmethod
    def execute_to_json(self, statement):
        pass

    @property
    def db_type(self):
        return self._db_type

    @property
    def desc(self):
        return self._desc

    def _is_str_or_nested_dict(self, value):
        if isinstance(value, str):
            return True
        elif isinstance(value, dict):
            return all(self._is_str_or_nested_dict(v) for v in value.values())
        return False

    def _validate_desc(self, d):
        return isinstance(d, dict) and all(self._is_str_or_nested_dict(v) for v in d.values())

    def _serialize_uncommon_type(self, obj):
        if not isinstance(obj, int, str, float, bool, tuple, list, dict):
            return str(obj)
