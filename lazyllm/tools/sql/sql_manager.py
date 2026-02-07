import json
import re
from contextlib import contextmanager
from typing import List, Type, Union, Dict, Any
from urllib.parse import quote_plus
import pydantic
import sqlalchemy
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import DeclarativeBase, DeclarativeMeta, sessionmaker

from .db_manager import DBManager, DBResult, DBStatus


class TableBase(DeclarativeBase):
    pass


class ColumnInfo(pydantic.BaseModel):
    name: str
    data_type: str
    comment: str = ''
    # At least one column should be True
    is_primary_key: bool = False
    nullable: bool = True
    default: Any = None


class TableInfo(pydantic.BaseModel):
    name: str
    comment: str = ''
    columns: list[ColumnInfo]


class TablesInfo(pydantic.BaseModel):
    tables: list[TableInfo]

class SqlManager(DBManager):
    DB_TYPE_SUPPORTED = set(['postgresql', 'mysql', 'mssql', 'sqlite', 'mysql+pymysql', 'tidb'])
    DB_DRIVER_MAP = {'mysql': 'pymysql', 'tidb': 'pymysql'}
    PYTYPE_TO_SQL_MAP = {
        'integer': sqlalchemy.Integer,
        'string': sqlalchemy.Text,
        'text': sqlalchemy.Text,
        'boolean': sqlalchemy.Boolean,
        'float': sqlalchemy.Float,
        'datetime': sqlalchemy.DateTime,
        'bytes': sqlalchemy.LargeBinary,
        'bool': sqlalchemy.Boolean,
        'date': sqlalchemy.Date,
        'time': sqlalchemy.Time,
        'list': sqlalchemy.ARRAY,
        'dict': sqlalchemy.JSON,
        'uuid': sqlalchemy.Uuid,
    }

    def __init__(self, db_type: str, user: str, password: str, host: str, port: int, db_name: str, *,
                 options_str: str = None, tables_info_dict: Dict = None):
        db_type = db_type.lower()
        if db_type not in self.DB_TYPE_SUPPORTED:
            raise ValueError(f'{db_type} not supported')
        super().__init__(db_type)
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._db_name = db_name
        self._tables_desc_dict = {}
        self._visible_tables = None
        self._metadata = sqlalchemy.MetaData()
        self._options_str = options_str
        self._orm_cache = {}
        self._engine = None
        self._Session = None
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        if tables_info_dict:
            self._init_tables_by_info(tables_info_dict)

    def _init_tables_by_info(self, tables_info_dict):
        try:
            tables_info = TablesInfo.model_validate(tables_info_dict)
            self._visible_tables = [table_info.name for table_info in tables_info.tables]
            # create table if not exist
            self._create_tables_by_info(tables_info)
            desc_dict = self._gen_desc_by_info(tables_info)
            self.set_desc(desc_dict)
        except pydantic.ValidationError as e:
            raise ValueError(f'Validate tables_info_dict failed: {str(e)}')

    def _sql_type_for(self, py_type: str):
        t = py_type.lower()
        if self._db_type in ('mysql', 'tidb', 'mysql+pymysql'):
            if t == 'list':
                return sqlalchemy.JSON
            if t == 'uuid':
                return sqlalchemy.String(36)
        return self.PYTYPE_TO_SQL_MAP.get(t, sqlalchemy.Text)

    def _create_tables_by_info(self, tables_info: TablesInfo):
        for table_info in tables_info.tables:
            attrs = {'__tablename__': table_info.name, '__table_args__': {'extend_existing': True},
                     'metadata': self._metadata}
            for column_info in table_info.columns:
                column_type = column_info.data_type.lower()
                is_nullable = column_info.nullable
                column_name = column_info.name
                is_primary = column_info.is_primary_key
                default_value = column_info.default
                real_type = self._sql_type_for(column_type)
                # Handle default value
                if default_value is not None:
                    attrs[column_name] = sqlalchemy.Column(real_type, nullable=is_nullable,
                                                           primary_key=is_primary, default=default_value)
                else:
                    attrs[column_name] = sqlalchemy.Column(real_type, nullable=is_nullable, primary_key=is_primary)
            # When create dynamic class with same name, old version will be replaced
            TableClass = type(table_info.name.capitalize(), (TableBase,), attrs)
            self.create_table(TableClass)

    def _gen_desc_by_info(self, tables_info: TablesInfo) -> dict:
        desc_dict = {}
        for table_info in tables_info.tables:
            table_comment = ''
            if table_info.comment:
                table_comment += f'COMMENT ON TABLE "{table_info.name}": {table_info.comment}\n'
            for column_info in table_info.columns:
                table_comment += f'COMMENT ON COLUMN "{table_info.name}.{column_info.name}": {column_info.comment}\n'
            if table_comment:
                desc_dict[table_info.name] = table_comment
        return desc_dict

    def _gen_conn_url(self) -> str:
        if self._db_type == 'sqlite':
            conn_url = f'sqlite:///{self._db_name}{("?" + self._options_str) if self._options_str else ""}'
        else:
            driver = self.DB_DRIVER_MAP.get(self._db_type if self._db_type != 'tidb' else 'mysql', '')
            password = quote_plus(self._password)
            prefix = 'mysql' if self._db_type == 'tidb' else self._db_type
            conn_url = (f'{prefix}{("+" + driver) if driver else ""}://{self._user}:{password}@{self._host}'
                        f':{self._port}/{self._db_name}{("?" + self._options_str) if self._options_str else ""}')
        return conn_url

    @property
    def engine(self):
        if self._engine is None:
            conn_url = self._gen_conn_url()
            if self._db_type == 'sqlite':
                self._engine = sqlalchemy.create_engine(
                    conn_url,
                    connect_args={'check_same_thread': False, 'timeout': 30},
                    poolclass=sqlalchemy.pool.QueuePool,
                    echo=False
                )
                with self._engine.connect() as conn:
                    conn.execute(sqlalchemy.text('PRAGMA journal_mode=WAL'))
                    conn.execute(sqlalchemy.text('PRAGMA synchronous=NORMAL'))
                    conn.execute(sqlalchemy.text('PRAGMA busy_timeout=30000'))
                    conn.commit()
            elif self._db_type == 'tidb':
                self._engine = sqlalchemy.create_engine(
                    conn_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    connect_args={},
                    echo=False,
                )
            else:
                self._engine = sqlalchemy.create_engine(
                    conn_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
        return self._engine

    @property
    def Session(self):
        if self._Session is None:
            self._Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        return self._Session

    @contextmanager
    def get_session(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def check_connection(self) -> DBResult:
        try:
            with self.engine.connect() as _:
                return DBResult()
        except SQLAlchemyError as e:
            return DBResult(status=DBStatus.FAIL, detail=str(e))

    @property
    def desc(self) -> str:
        if self._desc is None:
            self.set_desc(tables_desc_dict={})
        return self._desc

    def set_desc(self, tables_desc_dict: dict = {}):  # noqa B006
        self._desc = ''
        if not isinstance(tables_desc_dict, dict):
            raise ValueError(f'desc type {type(tables_desc_dict)} not supported')
        self._tables_desc_dict = tables_desc_dict
        if len(self.visible_tables) == 0:
            return
        # Generate desc according to table schema and comment
        self._desc = 'The tables description is as follows\n```\n'
        for table_name in self.visible_tables:
            self._desc += f'Table {table_name}\n(\n'
            TableCls = self.get_table_orm_class(table_name)
            if TableCls is None:
                # The table could be dropped in other session
                continue
            table_columns = TableCls.__table__.columns
            for i, column in enumerate(table_columns):
                self._desc += f' {column.name} {column.type}'
                if i != len(table_columns) - 1:
                    self._desc += ','
                self._desc += '\n'
            self._desc += ');\n'
            if table_name in tables_desc_dict:
                self._desc += tables_desc_dict[table_name] + '\n\n'
        self._desc += '```\n'

    @property
    def visible_tables(self):
        if self._visible_tables is None:
            self._visible_tables = self.get_all_tables()
        return self._visible_tables

    @visible_tables.setter
    def visible_tables(self, visible_tables: list):
        all_tables = set(self.get_all_tables())
        for ele in visible_tables:
            if ele not in all_tables:
                raise ValueError(f'Table {ele} not found in database')
        self._visible_tables = visible_tables
        self.set_desc(self._tables_desc_dict)

    def _refresh_metadata(self, only=None):
        # refresh metadata in case of deleting/creating table in other session
        try:
            if only:
                self._metadata.reflect(bind=self.engine, only=only, extend_existing=True)
            elif not self._metadata.tables:
                self._metadata.reflect(bind=self.engine)
        except Exception as e:
            raise ValueError(f'Refresh metadata failed: {e}')

    def get_all_tables(self) -> list:
        self._refresh_metadata()
        return list(self._metadata.tables.keys())

    def get_table_orm_class(self, table_name):
        if table_name in self._orm_cache:
            return self._orm_cache[table_name]
        self._refresh_metadata(only=[table_name])
        Base = automap_base(metadata=self._metadata)
        Base.prepare()
        class_obj = getattr(Base.classes, table_name, None)
        self._orm_cache[table_name] = class_obj
        return class_obj

    def execute_commit(self, statement: str):
        with self.get_session() as session:
            session.execute(sqlalchemy.text(statement))

    def execute_query(self, statement: str) -> str:
        statement = re.sub(r'/\*.*?\*/', '', statement, flags=re.DOTALL).strip()
        create_table_pattern = r'.*\s*create\s+table\s+.*'
        drop_table_pattern = r'.*\s*drop\s+table\s+.*'
        statement_lower = statement.lower()
        if re.match(create_table_pattern, statement_lower):
            return f'Create table not supported. Original statement: {statement}'
        elif re.match(drop_table_pattern, statement_lower):
            return f'Drop table not supported. Original statement: {statement}'
        try:
            result = []
            session = self.Session()
            # Use original session without post commit
            with session as session:
                cursor_result = session.execute(sqlalchemy.text(statement))
                columns = list(cursor_result.keys())
                result = [dict(zip(columns, row)) for row in cursor_result]
            str_result = json.dumps(result, ensure_ascii=False, default=self._serialize_uncommon_type)
        except Exception as e:
            str_result = f'Execute SQL ERROR: {str(e)}'
        return str_result

    def _create_by_script(self, table: str) -> DBResult:
        status = DBStatus.SUCCESS
        detail = 'Success'
        try:
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(table))
                conn.commit()
        except OperationalError as e:
            status = DBStatus.FAIL
            detail = f'ERROR: {str(e)}'
        return DBResult(status=status, detail=detail)

    def _create_by_api(self, table: Union[DeclarativeBase, DeclarativeMeta]) -> DBResult:
        table.metadata.create_all(bind=self.engine, checkfirst=True)
        return DBResult()

    def create_table(self, table: Union[str, Type[DeclarativeBase], DeclarativeMeta]) -> DBResult:
        status = DBStatus.SUCCESS
        detail = 'Success'
        if isinstance(table, str):
            return self._create_by_script(table)
        # Support DeclarativeMeta created by declarative_base() which is deprecated since: 2.0
        elif issubclass(table, DeclarativeBase) or isinstance(table, DeclarativeMeta):
            return self._create_by_api(table)
        else:
            status = DBStatus.FAIL
            detail += f'Failed: Unsupported Type: {table}'
        return DBResult(status=status, detail=detail)

    def drop_table(self, table: Union[str, Type[DeclarativeBase], DeclarativeMeta]) -> DBResult:
        metadata = self._metadata
        if isinstance(table, str):
            tablename = table
        elif issubclass(table, DeclarativeBase) or isinstance(table, DeclarativeMeta):
            tablename = table.__tablename__
        else:
            return DBResult(status=DBStatus.FAIL, detail=f'{table} type unsupported')
        Table = sqlalchemy.Table(tablename, metadata, autoload_with=self.engine)
        Table.drop(self.engine, checkfirst=True)
        return DBResult()

    def insert_values(self, table_name: str, vals: List[dict]) -> DBResult:
        TableCls = self.get_table_orm_class(table_name)
        if TableCls is None:
            return DBResult(status=DBStatus.FAIL, detail=f'{table_name} not found in database')
        try:
            with self.get_session() as session:
                objects = [TableCls(**v) for v in vals]
                session.add_all(objects)
            return DBResult()
        except Exception as e:
            return DBResult(status=DBStatus.FAIL, detail=f'Insert failed: {e}')
