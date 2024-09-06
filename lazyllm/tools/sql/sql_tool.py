# flake8: noqa E501
from lazyllm.module import ModuleBase
import sqlite3
import lazyllm
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from lazyllm import pipeline, globals, bind, LOG, _0, switch
import json5 as json
from typing import List, Any, Dict, Union
from pathlib import Path
import datetime
import re
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.schema import CreateTable


class SqlTool:
    DB_TYPE_SUPPORTED = set(["PostgreSQL", "MySQL", "MS SQL"])

    def __init__(self, db_type: str, conn_url: str) -> None:
        self.reset_db(db_type, conn_url)

    def reset_db(self, db_type: str, conn_url: str):
        assert db_type in self.DB_TYPE_SUPPORTED
        extra_fields = {}
        if "?" in conn_url:
            npos = conn_url.find("?")
            str_extra = conn_url[npos + 1 :]
            extra_fields = {key: value for key_value in str_extra.split("&") for key, value in (key_value.split("="),)}
            conn_url = conn_url[0:npos]

        self.db_type = db_type
        self.conn_url = conn_url
        self.extra_fields = extra_fields
        self.engine = sqlalchemy.create_engine(conn_url)

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
                columns = result.keys()
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

    def get_tables_desc(self, tables: list = None) -> str:
        if tables is None or len(tables) == 0:
            tables = self.get_all_tables()
        metadata = sqlalchemy.MetaData()
        metadata.reflect(bind=self.engine)
        str_desc = ""
        for table_name in tables:
            cur_table = sqlalchemy.Table(table_name, metadata, autoload_with=self.engine)
            creation_sql = str(CreateTable(cur_table).compile(self.engine))
            str_desc += creation_sql + "\n"
        return str_desc

    def get_table_columns(self, table_name: str):
        inspector = sqlalchemy.inspect(self.engine)
        columns = inspector.get_columns(table_name, schema=self.extra_fields.get("schema", None))
        return columns


class SQLiteTool(SqlTool):
    def __init__(self, db_file, return_trace=False):
        super().__init__()
        self.db_type = ""
        assert Path(db_file).is_file()
        self._return_trace = return_trace
        self.conn = sqlite3.connect(db_file, check_same_thread=False)

    def __del__(self):
        self.close_connection()

    def create_tables(self, tables_info: dict):
        cursor = self.conn.cursor()
        for table_name, table_info in tables_info.items():
            # Start building the SQL for creating the table
            create_table_sql = f"CREATE TABLE {table_name} ("

            # Iterate over fields to add them to the SQL statement
            fields = []
            for field_name, field_info in table_info["fields"].items():
                field_type = field_info["type"]
                comment = field_info["comment"]

                # Add field definition
                fields.append(f"{field_name} {field_type} comment '{comment}'")

            # Join fields and complete SQL statement
            create_table_sql += ", ".join(fields) + ");"

            # Execute SQL statement to create the table
            cursor.execute(create_table_sql)
        cursor.close()
        self.conn.commit()

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def get_tables_desc(self, tables: list = None) -> str:
        sql_script = "SELECT sql FROM sqlite_master WHERE type='table'"
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql_script)
            table_infos = cursor.fetchall()
            str_tables = ""
            for table_info in table_infos:
                str_tables += table_info[0] + "\n"
            cursor.close()
            return str_tables
        except Exception as e:
            cursor.close()
            if self._return_trace:
                globals["trace"].append(f"SQLiteTool Exception: {str(e)}. sql_script: {sql_script}")
            LOG.warning(str(e))
            return ""

    def get_query_result_in_json(self, sql_script):
        cursor = self.conn.cursor()
        str_result = ""
        try:
            cursor.execute(sql_script)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            # change result to json
            results = [dict(zip(columns, row)) for row in rows]
            str_result = json.dumps(results, ensure_ascii=False)
        except sqlite3.Error as e:
            lazyllm.LOG.warning(f"SQLite error: {str(e)}")
            if self._return_trace:
                globals["trace"].append(f"SQLiteTool Exception: {str(e)}. sql_script: {sql_script}")
        finally:
            cursor.close()
        return str_result

    def sql_update(self, sql_script):
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql_script)
            # For INSERT, UPDATE execution must be committed
            self.conn.commit()
            cursor.close()
        except sqlite3.Error as e:
            lazyllm.LOG.warning(f"SQLite error: {str(e)}")
            if self._return_trace:
                globals["trace"].append(f"SQLiteTool Exception: {str(e)}. sql_script: {sql_script}")
        finally:
            cursor.close()


sql_query_instruct_template = """
Given the following SQL tables and current date {current_date}, your job is to write sql queries in {db_type} given a user’s request.
Alert: Just replay the sql query in a code block.

{sql_tables}
"""

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


class SqlModule(ModuleBase):
    def __init__(
        self,
        llm,
        sql_tool: SqlTool,
        tables: Union[List[str], None] = None,
        tables_desc: str = "",
        sql_examples: str = "",
        use_llm_for_sql_result=True,
        return_trace: bool = False,
    ) -> None:
        super().__init__(return_trace=return_trace)
        self._sql_tool = sql_tool
        self._query_prompter = ChatPrompter(instruction=sql_query_instruct_template).pre_hook(self.sql_query_promt_hook)
        self._llm_query = llm.share(prompt=self._query_prompter)
        self._answer_prompter = ChatPrompter(instruction=sql_explain_instruct_template).pre_hook(
            self.sql_explain_prompt_hook
        )
        self._llm_answer = llm.share(prompt=self._answer_prompter)
        self._pattern = re.compile(r"```sql(.+?)```", re.DOTALL)
        with pipeline() as sql_execute_ppl:
            sql_execute_ppl.exec = self._sql_tool.get_query_result_in_json
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
        globals._data.pop("root_input")
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
