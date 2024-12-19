from enum import Enum, unique
from typing import List, Union, overload
from pydantic import BaseModel
from abc import ABC, abstractmethod
from urllib.parse import quote_plus


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
        password = quote_plus(password)
        self.status = DBStatus.SUCCESS
        self.detail = ""
        db_type = db_type.lower()
        db_result = self.reset_engine(db_type, user, password, host, port, db_name, options_str)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(db_result.detail)

    @overload
    def reset_engine(self, db_type, user, password, host, port, db_name, options_str) -> DBResult:
        pass

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
