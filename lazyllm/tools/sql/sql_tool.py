from lazyllm.module import ModuleBase
import lazyllm
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from lazyllm import pipeline, globals, bind, _0, switch
import json
from typing import List, Any, Dict, Union
import datetime
import re
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import declarative_base
import pydantic
from urllib.parse import quote_plus


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


class SqlManager(ModuleBase):
    DB_TYPE_SUPPORTED = set(["PostgreSQL", "MySQL", "MSSQL", "SQLite"])
    SUPPORTED_DATA_TYPES = {
        "integer": sqlalchemy.Integer,
        "string": sqlalchemy.String,
        "boolean": sqlalchemy.Boolean,
        "float": sqlalchemy.Float,
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
        super().__init__()
        if db_type.lower() != "sqlite":
            password = quote_plus(password)
            conn_url = f"{db_type.lower()}://{user}:{password}@{host}:{port}/{db_name}"
            self.reset_db(db_type, conn_url, tables_info_dict, options_str)

    def forward(self, sql_script: str) -> str:
        return self.get_query_result_in_json(sql_script)

    def reset_tables(self, tables_info_dict: dict) -> tuple[bool, str]:
        existing_tables = set(self.get_all_tables())
        try:
            tables_info = TablesInfo.model_validate(tables_info_dict)
        except pydantic.ValidationError as e:
            lazyllm.LOG.warning(str(e))
            return False, str(e)
        for table_info in tables_info.tables:
            if table_info.name not in existing_tables:
                # create table
                cur_rt, cur_err_msg = self._create_table(table_info.model_dump())
            else:
                # check table
                cur_rt, cur_err_msg = self._check_columns_match(table_info.model_dump())
            if not cur_rt:
                lazyllm.LOG.warning(f"cur_err_msg: {cur_err_msg}")
                return cur_rt, cur_err_msg
        rt, err_msg = self._set_tables_desc_prompt(tables_info_dict)
        if not rt:
            lazyllm.LOG.warning(err_msg)
        return True, "Success"

    def reset_db(self, db_type: str, conn_url: str, tables_info_dict: dict, options_str=""):
        assert db_type in self.DB_TYPE_SUPPORTED
        extra_fields = {}
        if options_str:
            extra_fields = {
                key: value for key_value in options_str.split("&") for key, value in (key_value.split("="),)
            }
        self.db_type = db_type
        self.conn_url = conn_url
        self.extra_fields = extra_fields
        self.engine = sqlalchemy.create_engine(conn_url)
        self.tables_prompt = ""
        rt, err_msg = self.reset_tables(tables_info_dict)
        if not rt:
            self.err_msg = err_msg
            self.err_code = 1001
        else:
            self.err_code = 0

    def get_tables_desc(self):
        return self.tables_prompt

    def check_connection(self) -> tuple[bool, str]:
        try:
            with self.engine.connect() as _:
                return True, "Success"
        except SQLAlchemyError as e:
            return False, str(e)

    def get_query_result_in_json(self, sql_script) -> str:
        str_result = ""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sqlalchemy.text(sql_script))
                columns = list(result.keys())
                result_dict = [dict(zip(columns, row)) for row in result]
                str_result = json.dumps(result_dict, ensure_ascii=False)
        except OperationalError as e:
            str_result = f"ERROR: {str(e)}"
        finally:
            if "conn" in locals():
                conn.close()
        return str_result

    def get_all_tables(self) -> list:
        inspector = sqlalchemy.inspect(self.engine)
        table_names = inspector.get_table_names(schema=self.extra_fields.get("schema", None))
        return table_names

    def execute_sql_update(self, sql_script):
        rt, err_msg = True, "Success"
        try:
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(sql_script))
                conn.commit()
        except OperationalError as e:
            lazyllm.LOG.warning(f"sql error: {str(e)}")
            rt, err_msg = False, str(e)
        finally:
            if "conn" in locals():
                conn.close()
        return rt, err_msg

    def _get_table_columns(self, table_name: str):
        inspector = sqlalchemy.inspect(self.engine)
        columns = inspector.get_columns(table_name, schema=self.extra_fields.get("schema", None))
        return columns

    def _create_table(self, table_info_dict: dict) -> tuple[bool, str]:
        rt, err_msg = True, "Success"
        try:
            table_info = TableInfo.model_validate(table_info_dict)
        except pydantic.ValidationError as e:
            return False, str(e)
        try:
            with self.engine.connect() as conn:
                Base = declarative_base()
                # build table dynamically
                attrs = {"__tablename__": table_info.name}
                for column_info in table_info.columns:
                    column_type = column_info.data_type.lower()
                    is_nullable = column_info.nullable
                    column_name = column_info.name
                    is_primary = column_info.is_primary_key
                    if column_type not in self.SUPPORTED_DATA_TYPES:
                        return False, f"Unsupported column type: {column_type}"
                    real_type = self.SUPPORTED_DATA_TYPES[column_type]
                    attrs[column_name] = sqlalchemy.Column(real_type, nullable=is_nullable, primary_key=is_primary)
                TableClass = type(table_info.name.capitalize(), (Base,), attrs)
                Base.metadata.create_all(self.engine)
        except OperationalError as e:
            rt, err_msg = False, f"ERROR: {str(e)}"
        finally:
            if "conn" in locals():
                conn.close()
        return rt, err_msg

    def _delete_rows_by_name(self, table_name):
        metadata = sqlalchemy.MetaData()
        metadata.reflect(bind=self.engine)
        rt, err_msg = True, "Success"
        try:
            with self.engine.connect() as conn:
                table = sqlalchemy.Table(table_name, metadata, autoload_with=self.engine)
                delete = table.delete()
                conn.execute(delete)
                conn.commit()
        except SQLAlchemyError as e:
            rt, err_msg = False, str(e)
        return rt, err_msg

    def _drop_table_by_name(self, table_name):
        metadata = sqlalchemy.MetaData()
        metadata.reflect(bind=self.engine)
        rt, err_msg = True, "Success"
        try:
            table = sqlalchemy.Table(table_name, metadata, autoload_with=self.engine)
            table.drop(bind=self.engine, checkfirst=True)
        except SQLAlchemyError as e:
            lazyllm.LOG.warning("GET SQLAlchemyError")
            rt, err_msg = False, str(e)
        return rt, err_msg

    def _check_columns_match(self, table_info_dict: dict):
        try:
            table_info = TableInfo.model_validate(table_info_dict)
        except pydantic.ValidationError as e:
            return False, str(e)
        real_columns = self._get_table_columns(table_info.name)
        tmp_dict = {}
        for real_column in real_columns:
            tmp_dict[real_column["name"]] = (real_column["type"], real_column["nullable"])
        for column_info in table_info.columns:
            if column_info.name not in tmp_dict:
                return False, f"Table {table_info.name} exists but column {column_info.name} does not."
            real_column = tmp_dict[column_info.name]
            column_type = column_info.data_type.lower()
            if column_type not in self.SUPPORTED_DATA_TYPES:
                return False, f"Unsupported column type: {column_type}"
            # 1. check data type
            # string type sometimes changes to other type (such as varchar)
            real_type_cls = real_column[0].__class__
            if column_type != real_type_cls.__name__.lower() and not issubclass(
                real_type_cls, self.SUPPORTED_DATA_TYPES[column_type]
            ):
                return (
                    False,
                    f"Table {table_info.name} exists but column {column_info.name} data_type mismatch"
                    f": {column_info.data_type} vs {real_column[0].__class__.__name__}",
                )
            # 2. check nullable
            if column_info.nullable != real_column[1]:
                return False, f"Table {table_info.name} exists but column {column_info.name} nullable mismatch"
        if len(tmp_dict) > len(table_info.columns):
            return (
                False,
                f"Table {table_info.name} exists but has more columns. {len(tmp_dict)} vs {len(table_info.columns)}",
            )
        return True, "Match"

    def _set_tables_desc_prompt(self, tables_info_dict: dict) -> str:
        try:
            tables_info = TablesInfo.model_validate(tables_info_dict)
        except pydantic.ValidationError as e:
            return False, str(e)
        self.tables_prompt = "The tables description is as follows\n```\n"
        for table_info in tables_info.tables:
            self.tables_prompt += f'Table "{table_info.name}"'
            if table_info.comment:
                self.tables_prompt += f' comment "{table_info.comment}"'
            self.tables_prompt += "\n(\n"
            for i, column_info in enumerate(table_info.columns):
                self.tables_prompt += f"{column_info.name} {column_info.data_type}"
                if column_info.comment:
                    self.tables_prompt += f' comment "{column_info.comment}"'
                if i != len(table_info.columns) - 1:
                    self.tables_prompt += ","
                self.tables_prompt += "\n"
            self.tables_prompt += ");\n"
        self.tables_prompt += "```\n"
        return True, "Success"


class SQLiteManger(SqlManager):

    def __init__(self, db_file, tables_info_dict: dict):
        super().__init__("SQLite", "", "", "", 0, "", {}, "")
        super().reset_db("SQLite", f"sqlite:///{db_file}", tables_info_dict)


sql_query_instruct_template = """
Given the following SQL tables and current date {current_date}, your job is to write sql queries in {db_type} given a user’s request.
Alert: Just replay the sql query in a code block.

{sql_tables}
"""  # noqa E501

sql_explain_instruct_template = """
According to chat history
```
{history_info}
```

bellowing sql query is executed

```
{sql_query}
```
the sql result is
```
{sql_result}
```
"""


class SqlCall(ModuleBase):
    def __init__(
        self,
        llm,
        sql_manager: SqlManager,
        sql_examples: str = "",
        use_llm_for_sql_result=True,
        return_trace: bool = False,
    ) -> None:
        super().__init__(return_trace=return_trace)
        self._sql_tool = sql_manager
        self._query_prompter = ChatPrompter(instruction=sql_query_instruct_template).pre_hook(self.sql_query_promt_hook)
        self._llm_query = llm.share(prompt=self._query_prompter).used_by(self._module_id)
        self._answer_prompter = ChatPrompter(instruction=sql_explain_instruct_template).pre_hook(
            self.sql_explain_prompt_hook
        )
        self._llm_answer = llm.share(prompt=self._answer_prompter).used_by(self._module_id)
        self._pattern = re.compile(r"```sql(.+?)```", re.DOTALL)
        with pipeline() as sql_execute_ppl:
            sql_execute_ppl.exec = self._sql_tool
            if use_llm_for_sql_result:
                sql_execute_ppl.concate = (lambda q, r: [q, r]) | bind(sql_execute_ppl.input, _0)
                sql_execute_ppl.llm_answer = self._llm_answer
        with pipeline() as ppl:
            ppl.llm_query = self._llm_query
            ppl.sql_extractor = self.extract_sql_from_response
            with switch(judge_on_full_input=False) as ppl.sw:
                ppl.sw.case[False, lambda x: x]
                ppl.sw.case[True, sql_execute_ppl]
        self._impl = ppl

    def sql_query_promt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        sql_tables_info = self._sql_tool.get_tables_desc()
        if not isinstance(input, str):
            raise ValueError(f"Unexpected type for input: {type(input)}")
        return (
            dict(
                current_date=current_date, db_type=self._sql_tool.db_type, sql_tables=sql_tables_info, user_query=input
            ),
            history,
            tools,
            label,
        )

    def sql_explain_prompt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        explain_query = "Tell the user based on the sql execution results, making sure to keep the language consistent \
            with the user's input and don't translate original result."
        if not isinstance(input, list) and len(input) != 2:
            raise ValueError(f"Unexpected type for input: {type(input)}")
        assert "root_input" in globals and self._llm_answer._module_id in globals["root_input"]
        user_query = globals["root_input"][self._llm_answer._module_id]
        globals.pop("root_input")
        history_info = chat_history_to_str(history, user_query)
        return (
            dict(history_info=history_info, sql_query=input[0], sql_result=input[1], explain_query=explain_query),
            history,
            tools,
            label,
        )

    def extract_sql_from_response(self, str_response: str) -> tuple[bool, str]:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            return True, extracted_content
        else:
            return False, str_response

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        globals["root_input"] = {self._llm_answer._module_id: input}
        if self._module_id in globals["chat_history"]:
            globals["chat_history"][self._llm_query._module_id] = globals["chat_history"][self._module_id]
        return self._impl(input)
