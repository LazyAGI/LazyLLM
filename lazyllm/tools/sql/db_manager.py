from enum import Enum, unique
from typing import List, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod
from lazyllm.module import ModuleBase


@unique
class DBStatus(Enum):
    SUCCESS = 0
    FAIL = 1


class DBResult(BaseModel):
    status: DBStatus = DBStatus.SUCCESS
    detail: str = "Success"
    result: Union[List, None] = None

class CommonMeta(type(ABC), type(ModuleBase)):
    pass

class DBManager(ABC, ModuleBase, metaclass=CommonMeta):
    DB_TYPE_SUPPORTED = set(["postgresql", "mysql", "mssql", "sqlite", "mongodb"])

    def __init__(self, db_type: str):
        db_type = db_type.lower()
        ModuleBase.__init__(self)
        if db_type not in self.DB_TYPE_SUPPORTED:
            raise ValueError(f"{db_type} not supported")
        self._db_type = db_type
        self._desc = None

    @abstractmethod
    def execute_query(self, statement) -> str:
        pass

    def forward(self, statement: str) -> str:
        return self.execute_query(statement)

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
