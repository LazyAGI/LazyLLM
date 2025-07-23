from enum import Enum, unique
from typing import List, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod
from lazyllm.module import ModuleBase


@unique
class DBStatus(Enum):
    """An enumeration."""
    SUCCESS = 0
    FAIL = 1


class DBResult(BaseModel):
    status: DBStatus = DBStatus.SUCCESS
    detail: str = "Success"
    result: Union[List, None] = None

class CommonMeta(type(ABC), type(ModuleBase)):
    pass

class DBManager(ABC, ModuleBase, metaclass=CommonMeta):

    def __init__(self, db_type: str):
        ModuleBase.__init__(self)
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
        return all(isinstance(key, str) and (isinstance(value, str) or DBManager._is_dict_all_str(value))
                   for key, value in d.items())

    @staticmethod
    def _serialize_uncommon_type(obj):
        if not isinstance(obj, (int, str, float, bool, tuple, list, dict)):
            return str(obj)
