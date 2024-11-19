import json
from typing import Union
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from sqlalchemy.orm import declarative_base, DeclarativeMeta
import pydantic
from .db_manager import DBManager, DBStatus, DBResult
from pathlib import Path


class SqlManagerBase(DBManager):

    def __init__(self, db_type, user, password, host, port, db_name, options_str="", set_default_des=True):
        self._set_default_desc = set_default_des
        super().__init__(db_type, user, password, host, port, db_name, options_str)

    def reset_engine(
        self,
        db_type: str,
        user: str,
        password: str,
        host: str,
        port: int,
        db_name: str,
        options_str: str = "",
    ):
        self._db_type = db_type
        if db_type not in self.DB_TYPE_SUPPORTED:
            return DBResult(status=DBStatus.FAIL, detail=f"{db_type} not supported")
        if db_type in self.DB_DRIVER_MAP:
            conn_url = f"{db_type}+{self.DB_DRIVER_MAP[db_type]}://{user}:{password}@{host}:{port}/{db_name}"
        else:
            conn_url = f"{db_type}://{user}:{password}@{host}:{port}/{db_name}"
        self._conn_url = conn_url

        self._engine = sqlalchemy.create_engine(self._conn_url)
        self._desc = ""
        extra_fields = {}
        if options_str:
            extra_fields = {
                key: value for key_value in options_str.split("&") for key, value in (key_value.split("="),)
            }
        self._extra_fields = extra_fields
        db_result = self.check_connection()
        if db_result.status != DBStatus.SUCCESS:
            return db_result
        db_result = self.get_all_tables()
        if db_result.status != DBStatus.SUCCESS:
            return db_result
        self._visible_tables = db_result.result
        if self._set_default_desc:
            return self.set_desc()
        return db_result

    def get_all_tables(self) -> DBResult:
        inspector = sqlalchemy.inspect(self._engine)
        table_names = inspector.get_table_names(schema=self._extra_fields.get("schema", None))
        if self.status != DBStatus.SUCCESS:
            return DBResult(status=self.status, detail=self.detail, result=None)
        return DBResult(result=table_names)

    def check_connection(self) -> DBResult:
        try:
            with self._engine.connect() as _:
                return DBResult()
        except SQLAlchemyError as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    @property
    def visible_tables(self):
        return self._visible_tables

    def set_visible_tables(self, tables: list[str]) -> DBResult:
        db_result = self.get_all_tables()
        if db_result.status != DBStatus.SUCCESS:
            return db_result
        all_tables_in_db = set(db_result.result)
        visible_tables = []
        failed_tables = []
        for ele in tables:
            if ele in all_tables_in_db:
                visible_tables.append(ele)
            else:
                failed_tables.append(ele)
        if len(tables) != len(visible_tables):
            db_result = DBResult(status=DBStatus.FAIL, detail=f"{failed_tables} missing in database")
        else:
            db_result = DBResult()
            self._visible_tables = visible_tables
        return db_result

    def _get_table_columns(self, table_name: str):
        inspector = sqlalchemy.inspect(self._engine)
        columns = inspector.get_columns(table_name, schema=self._extra_fields.get("schema", None))
        return columns

    def set_desc(self, tables_desc: dict = {}) -> DBResult:
        self._desc = ""
        if not isinstance(tables_desc, dict):
            return DBResult(status=DBStatus.FAIL, detail=f"desc type {type(tables_desc)} not supported")
        if len(tables_desc) == 0:
            return DBResult(status=DBStatus.FAIL, detail="Empty desc")
        if len(self.visible_tables) == 0:
            return DBResult()
        self._desc = "The tables description is as follows\n```\n"
        for table_name in self.visible_tables:
            self._desc += f"Table {table_name}\n(\n"
            table_columns = self._get_table_columns(table_name)
            for i, column in enumerate(table_columns):
                self._desc += f" {column['name']} {column['type']}"
                if i != len(table_columns) - 1:
                    self._desc += ","
                self._desc += "\n"
            self._desc += ");\n"
            if table_name in tables_desc:
                self._desc += tables_desc[table_name] + "\n\n"
        self._desc += "```\n"
        return DBResult()

    @property
    def desc(self) -> str:
        return self._desc

    def execute(self, statement) -> DBResult:
        if isinstance(statement, str):
            statement = sqlalchemy.text(statement)
        if isinstance(
            statement,
            (sqlalchemy.TextClause, sqlalchemy.Select, sqlalchemy.Insert, sqlalchemy.Update, sqlalchemy.Delete),
        ):
            status = DBStatus.SUCCESS
            detail = ""
            result = None
            try:
                with self._engine.connect() as conn:
                    cursor_result = conn.execute(statement)
                    conn.commit()
                    if cursor_result.returns_rows:
                        columns = list(cursor_result.keys())
                        result = [dict(zip(columns, row)) for row in cursor_result]
            except OperationalError as e:
                status = DBStatus.FAIL
                detail = f"ERROR: {str(e)}"
            finally:
                if "conn" in locals():
                    conn.close()
            return DBResult(status=status, detail=detail, result=result)
        else:
            return DBResult(status=DBStatus.FAIL, detail="statement type not supported")

    def execute_to_json(self, statement) -> str:
        dbresult = self.execute(statement)
        if dbresult.status != DBStatus.SUCCESS:
            self.status, self.detail = dbresult.status, dbresult.detail
            return ""
        if dbresult.result is None:
            return ""
        str_result = json.dumps(dbresult.result, ensure_ascii=False, default=self._serialize_uncommon_type)
        return str_result

    def _create_by_script(self, table: str) -> DBResult:
        status = DBStatus.SUCCESS
        detail = "Success"
        try:
            with self._engine.connect() as conn:
                conn.execute(sqlalchemy.text(table))
                conn.commit()
        except OperationalError as e:
            status = DBStatus.FAIL
            detail = f"ERROR: {str(e)}"
        finally:
            if "conn" in locals():
                conn.close()
        return DBResult(status=status, detail=detail)

    def _match_exist_table(self, table: DeclarativeMeta) -> DBResult:
        status = DBStatus.SUCCESS
        detail = f"Table {table.__tablename__} already exists."
        metadata = sqlalchemy.MetaData()
        exist_table = sqlalchemy.Table(table.__tablename__, metadata, autoload_with=self._engine)
        if len(table.__table__.columns) != len(exist_table.columns):
            status = DBStatus.FAIL
            detail += (
                f"\n Column number mismatch: {len(table.__table__.columns)} VS " f"{len(exist_table.columns)}(exists)"
            )
            return DBResult(status=status, detail=detail)
        for exist_column in exist_table.columns:
            target_column = getattr(table, exist_column.name)
            exist_type = type(exist_column.type)
            target_type = type(target_column.type)
            type_is_subclass = issubclass(exist_type, target_type) or issubclass(target_type, exist_type)
            if target_type is not sqlalchemy.types.TypeEngine and not type_is_subclass:
                detail += f"type mismatch {exist_type}  vs {target_type}"
                return DBResult(status=DBStatus.FAIL, detail=detail)
            for attr in ["primary_key", "nullable"]:
                if getattr(exist_column, attr) != getattr(target_column, attr):
                    detail += f"{attr} mismatch {getattr(exist_column, attr)} vs {getattr(target_column, attr)}"
                    return DBResult(status=DBStatus.FAIL, detail=detail)
        return DBResult()

    def _create_by_api(self, table: DeclarativeMeta) -> DBResult:
        try:
            table.__table__.create(bind=self._engine)
            return DBResult()
        except ProgrammingError as e:
            if "already exists" in str(e):
                return self._match_exist_table(table)

    def create(self, table: Union[str, DeclarativeMeta]) -> DBResult:
        status = DBStatus.SUCCESS
        detail = "Success"
        if isinstance(table, str):
            return self._create_by_script(table)
        elif isinstance(table, DeclarativeMeta):
            return self._create_by_api(table)
        else:
            status = DBStatus.FAIL
            detail += "\n Unsupported Type: {table}"
        return DBResult(status=status, detail=detail)

    def drop(self, table) -> DBResult:
        metadata = sqlalchemy.MetaData()
        if isinstance(table, str):
            tablename = table
        elif isinstance(table, DeclarativeMeta):
            tablename = table.__tablename__
        else:
            return DBResult(status=DBStatus.FAIL, detail=f"{table} type unsupported")
        Table = sqlalchemy.Table(tablename, metadata, autoload_with=self._engine)
        Table.drop(self._engine, checkfirst=True)
        return DBResult()

    def insert(self, statement) -> DBResult:
        if isinstance(statement, (str, sqlalchemy.Insert)):
            return self.execute(statement)
        elif isinstance(statement, dict):
            table_name = statement.get("table_name", None)
            table_data = statement.get("table_data", [])
            returning = statement.get("returning", [])
            if not table_name:
                return DBResult(status=DBStatus.FAIL, detail="No table_name found")
            if not table_data:
                return DBResult(status=DBStatus.FAIL, detail="No table_data found")
            metadata = sqlalchemy.MetaData()
            table = sqlalchemy.Table(table_name, metadata, autoload_with=self._engine)
            if not returning:
                statement = sqlalchemy.insert(table).values(table_data)
            else:
                return_columns = [sqlalchemy.column(ele) for ele in returning]
                statement = (sqlalchemy.insert(table).values(table_data)).returning(*return_columns)
            return self.execute(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail="statement type not supported")

    def update(self, statement) -> DBResult:
        if isinstance(statement, (str, sqlalchemy.Update)):
            return self.execute(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail="statement type not supported")

    def delete(self, statement) -> DBResult:
        if isinstance(statement, (str, sqlalchemy.Delete)):
            if isinstance(statement, str):
                tmp = statement.rstrip()
                if len(tmp.split()) == 1:
                    statement = f"DELETE FROM {tmp}"
            return self.execute(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail="statement type not supported")

    def select(self, statement) -> DBResult:
        if isinstance(statement, (str, sqlalchemy.Select)):
            return self.execute(statement)
        else:
            return DBResult(status=DBStatus.FAIL, detail="statement type not supported")


class ColumnInfo(pydantic.BaseModel):
    name: str
    data_type: str
    comment: str = ""
    # At least one column should be True
    is_primary_key: bool = False
    nullable: bool = True


class TableInfo(pydantic.BaseModel):
    name: str
    comment: str = ""
    columns: list[ColumnInfo]


class TablesInfo(pydantic.BaseModel):
    tables: list[TableInfo]

class SqlManager(SqlManagerBase):
    PYTYPE_TO_SQL_MAP = {
        "integer": sqlalchemy.Integer,
        "string": sqlalchemy.Text,
        "text": sqlalchemy.Text,
        "boolean": sqlalchemy.Boolean,
        "float": sqlalchemy.Float,
        "datetime": sqlalchemy.DateTime,
        "bytes": sqlalchemy.LargeBinary,
        "bool": sqlalchemy.Boolean,
        "date": sqlalchemy.Date,
        "time": sqlalchemy.Time,
        "list": sqlalchemy.ARRAY,
        "dict": sqlalchemy.JSON,
        "uuid": sqlalchemy.Uuid,
        "any": sqlalchemy.types.TypeEngine,
    }

    def __init__(
        self,
        db_type: str,
        user: str,
        password: str,
        host: str,
        port: int,
        db_name: str,
        tables_info_dict: dict,
        options_str: str = "",
    ) -> None:
        self._tables_info_dict = tables_info_dict
        super().__init__(db_type, user, password, host, port, db_name, options_str, set_default_des=False)

    def reset_engine(self, db_type, user, password, host, port, db_name, options_str):
        super().reset_engine(db_type, user, password, host, port, db_name, options_str)
        db_result = self.reset_table_info_dict(self._tables_info_dict)
        self.status = db_result.status
        self.detail = db_result.detail
        if self.status != DBStatus.SUCCESS:
            raise ValueError(self.detail)
        return db_result

    def reset_table_info_dict(self, tables_info_dict: dict) -> DBResult:
        self.status = DBStatus.SUCCESS
        self.detail = "Success"
        self._tables_info_dict = tables_info_dict
        try:
            tables_info = TablesInfo.model_validate(self._tables_info_dict)
        except pydantic.ValidationError as e:
            self.status, self.detail = DBStatus.FAIL, str(e)
            return DBResult(status=DBStatus.FAIL, detail=str(e))
        # Create or Check tables
        created_tables = []
        for table_info in tables_info.tables:
            TableClass = self._create_table_cls(table_info)
            db_result = self.create(TableClass)
            if db_result.status != DBStatus.SUCCESS:
                # drop partial created table
                for created_table in created_tables:
                    self.drop(created_table)
                return db_result
            created_tables.append(TableClass)

        db_result = self.set_visible_tables([ele.__tablename__ for ele in created_tables])
        if db_result.status != DBStatus.SUCCESS:
            return db_result
        return self.set_desc()

    def _create_table_cls(self, table_info: TableInfo) -> DeclarativeMeta:
        Base = declarative_base()
        attrs = {"__tablename__": table_info.name}
        for column_info in table_info.columns:
            column_type = column_info.data_type.lower()
            is_nullable = column_info.nullable
            column_name = column_info.name
            is_primary = column_info.is_primary_key
            real_type = self.PYTYPE_TO_SQL_MAP[column_type]
            attrs[column_name] = sqlalchemy.Column(real_type, nullable=is_nullable, primary_key=is_primary)
        TableClass = type(table_info.name.capitalize(), (Base,), attrs)
        return TableClass

    def set_desc(self) -> DBResult:
        self._desc = ""
        try:
            tables_info = TablesInfo.model_validate(self._tables_info_dict)
        except pydantic.ValidationError as e:
            self.status, self.detail = DBStatus.FAIL, str(e)
            return DBResult(status=DBStatus.FAIL, detail=str(e))
        self._desc = "The tables description is as follows\n```\n"
        for table_info in tables_info.tables:
            self._desc += f'Table "{table_info.name}"'
            if table_info.comment:
                self._desc += f' comment "{table_info.comment}"'
            self._desc += "\n(\n"
            real_columns = self._get_table_columns(table_info.name)
            column_type_dict = {}
            for real_column in real_columns:
                column_type_dict[real_column["name"]] = real_column["type"]
            for i, column_info in enumerate(table_info.columns):
                self._desc += f"{column_info.name} {column_type_dict[column_info.name]}"
                if column_info.comment:
                    self._desc += f' comment "{column_info.comment}"'
                if i != len(table_info.columns) - 1:
                    self._desc += ","
                self._desc += "\n"
            self._desc += ");\n"
        self._desc += "```\n"
        return DBResult()


class SQLiteManger(SqlManager):

    def __init__(self, db_path: str, tables_info_dict: dict = {}):
        result = self.reset_engine(db_path, tables_info_dict)
        self.status, self.detail = result.status, result.detail
        if self.status != DBStatus.SUCCESS:
            raise ValueError(self.detail)

    def reset_engine(self, db_path: str, tables_info_dict: dict):
        self._db_type = "sqlite"
        self.status = DBStatus.SUCCESS
        self.detail = ""
        self._conn_url = f"sqlite:///{db_path}"
        self._extra_fields = {}
        self._engine = sqlalchemy.create_engine(self._conn_url)
        return self.reset_table_info_dict(tables_info_dict)
