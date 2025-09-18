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
    detail: str = 'Success'
    result: Union[List, None] = None

class CommonMeta(type(ABC), type(ModuleBase)):
    pass

class DBManager(ABC, ModuleBase, metaclass=CommonMeta):
    """Abstract base class for database managers.

This class defines the standard interface and helpers for building database connectors, including a required `execute_query` method and description property.

Args:
    db_type (str): Type identifier of the database (e.g., 'mysql', 'mongodb').


Examples:
    >>> from lazyllm.components import DBManager
    >>> class DummyDB(DBManager):
    ...     def __init__(self):
    ...         super().__init__(db_type="dummy")
    ...     def execute_query(self, statement):
    ...         return f"Executed: {statement}"
    ...     @property
    ...     def desc(self):
    ...         return "Dummy database for testing."
    >>> db = DummyDB()
    >>> print(db("SELECT * FROM test"))
    ... Executed: SELECT * FROM test
    """

    def __init__(self, db_type: str):
        ModuleBase.__init__(self)
        self._db_type = db_type
        self._desc = None

    @abstractmethod
    def execute_query(self, statement) -> str:
        """Abstract method for executing database query statements. This method needs to be implemented by specific database manager subclasses to execute various database operations.

Args:
    statement: The database query statement to execute, which can be SQL statements or other database-specific query languages

Features of this method:

- **Abstract Method**: Requires implementation of specific database operation logic in subclasses
- **Unified Interface**: Provides a unified query interface for different database types
- **Error Handling**: Subclass implementations should include appropriate error handling and status reporting
- **Result Formatting**: Returns formatted string results for subsequent processing

**Note**: This method is the core method of the database manager, and all specific database operations are executed through this method.

"""
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
