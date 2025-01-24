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

    def __init__(self, db_type: str):
        db_type = db_type.lower()
        if db_type not in self.DB_TYPE_SUPPORTED:
            raise ValueError(f"{db_type} not supported")
        self._db_type = db_type
        self._desc = None

    @abstractmethod
    def execute_to_json(self, statement) -> str:
        pass

    @property
    def db_type(self) -> str:
        return self._db_type

    @property
    @abstractmethod
    def desc(self) -> str: pass

    @staticmethod
    def _is_dict_all_str(d):
        if not isinstance(d, dict):
            return False
        for key, value in d.items():
            if not isinstance(key, str):
                return False
            if isinstance(value, dict):
                if not DBManager._is_dict_all_str(value):
                    return False
            elif not isinstance(value, str):
                return False
        return True

    @staticmethod
    def _serialize_uncommon_type(obj):
        if not isinstance(obj, int, str, float, bool, tuple, list, dict):
            return str(obj)
