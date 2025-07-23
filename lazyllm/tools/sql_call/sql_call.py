from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from lazyllm import pipeline, globals, bind, _0, switch
from typing import List, Any, Dict, Union, Callable
import datetime
import re
from lazyllm.tools.sql import DBManager

sql_query_instruct_template = """
Given the following SQL tables and current date {current_date}, your job is to write sql queries in {db_type} given a user’s request.

{desc}

Alert: Just reply the sql query in a code block start with triple-backticks and keyword "sql"
"""  # noqa E501


mongodb_query_instruct_template = """
Current date is {current_date}.
You are a seasoned expert with 10 years of experience in crafting NoSQL queries for {db_type}. 
I will provide a collection description in a specified format. 
Your task is to analyze the user_question, which follows certain guidelines, and generate a NoSQL MongoDB aggregation pipeline accordingly.

{desc}

Note: Please return the json pipeline in a code block start with triple-backticks and keyword "json".
"""  # noqa E501

db_explain_instruct_template = """
According to chat history
```
{{history_info}}
```

and the {db_type} database description
```
{{desc}}
```

bellowing {statement_type} is executed

```
{{query}}
```
the result is
```
{{result}}
```
"""


class SqlCall(ModuleBase):
    """SqlCall is a class that extends ModuleBase and provides an interface for generating and executing SQL queries using a language model (LLM).
It is designed to interact with a SQL database, extract SQL queries from LLM responses, execute those queries, and return results or explanations.

Arguments:
    llm: A language model to be used for generating and interpreting SQL queries and explanations.
    sql_manager (SqlManager): An instance of SqlManager that handles interaction with the SQL database.
    sql_examples (str, optional): An example of converting natural language represented by a JSON string into an SQL statement, formatted as: [{"Question": "Find the names of people in the same department as Smith", "Answer": "SELECT...;"}]
    use_llm_for_sql_result (bool, optional): Default is True. If set to False, the module will only output raw SQL results in JSON without further processing.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.


Examples:
        >>> # First, run SqlManager example
        >>> import lazyllm
        >>> from lazyllm.tools import SQLiteManger, SqlCall
        >>> sql_tool = SQLiteManger("personal.db")
        >>> sql_llm = lazyllm.OnlineChatModule(model="gpt-4o", source="openai", base_url="***")
        >>> sql_call = SqlCall(sql_llm, sql_tool, use_llm_for_sql_result=True)
        >>> print(sql_call("去年一整年销售额最多的员工是谁?"))
    """
    EXAMPLE_TITLE = "Here are some example: "

    def __init__(
        self,
        llm,
        sql_manager: DBManager,
        sql_examples: str = "",
        sql_post_func: Callable = None,
        use_llm_for_sql_result=True,
        return_trace: bool = False,
    ) -> None:
        super().__init__(return_trace=return_trace)
        if not sql_manager.desc:
            raise ValueError("Error: sql_manager found empty description.")
        self._sql_tool = sql_manager
        self.sql_post_func = sql_post_func

        if sql_manager.db_type == "mongodb":
            self._query_prompter = ChatPrompter(instruction=mongodb_query_instruct_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = "mongodb json pipeline"
            self._pattern = re.compile(r"```json(.+?)```", re.DOTALL)
        else:
            self._query_prompter = ChatPrompter(instruction=sql_query_instruct_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = "sql query"
            self._pattern = re.compile(r"```sql(.+?)```", re.DOTALL)

        self._llm_query = llm.share(prompt=self._query_prompter).used_by(self._module_id)
        self._answer_prompter = ChatPrompter(
            instruction=db_explain_instruct_template.format(statement_type=statement_type, db_type=sql_manager.db_type)
        ).pre_hook(self.sql_explain_prompt_hook)
        self._llm_answer = llm.share(prompt=self._answer_prompter).used_by(self._module_id)
        self.example = sql_examples
        with pipeline() as sql_execute_ppl:
            sql_execute_ppl.exec = self._sql_tool.execute_query
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
        schema_desc = self._sql_tool.desc
        if self.example:
            schema_desc += f"\n{self.EXAMPLE_TITLE}\n{self.example}\n"
        if not isinstance(input, str):
            raise ValueError(f"Unexpected type for input: {type(input)}")
        return (
            dict(current_date=current_date, db_type=self._sql_tool.db_type, desc=schema_desc, user_query=input),
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
        explain_query = "Tell the user based on the execution results, making sure to keep the language consistent \
            with the user's input and don't translate original result."
        if not isinstance(input, list) and len(input) != 2:
            raise ValueError(f"Unexpected type for input: {type(input)}")
        assert "root_input" in globals and self._llm_answer._module_id in globals["root_input"]
        user_query = globals["root_input"][self._llm_answer._module_id]
        globals.pop("root_input")
        history_info = chat_history_to_str(history, user_query)
        return (
            dict(
                history_info=history_info,
                desc=self._sql_tool.desc,
                query=input[0],
                result=input[1],
                explain_query=explain_query,
            ),
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
            return True, extracted_content if not self.sql_post_func else self.sql_post_func(extracted_content)
        else:
            return False, str_response

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        globals["root_input"] = {self._llm_answer._module_id: input}
        if self._module_id in globals["chat_history"]:
            globals["chat_history"][self._llm_query._module_id] = globals["chat_history"][self._module_id]
        return self._impl(input)
