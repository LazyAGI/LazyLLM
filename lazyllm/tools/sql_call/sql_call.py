from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
from lazyllm.tools.utils import chat_history_to_str
from lazyllm import pipeline, globals, bind, _0, switch
from lazyllm.prompts import SqlCallPrompts
from typing import List, Any, Dict, Optional, Union, Callable
import datetime
import re
from lazyllm.tools.sql import DBManager


class SqlCall(ModuleBase):
    EXAMPLE_TITLE = "Here are some example: "

    def __init__(
        self,
        llm,
        sql_manager: DBManager,
        sql_examples: str = "",
        sql_post_func: Callable = None,
        use_llm_for_sql_result=True,
        prompts: Optional[SqlCallPrompts] = None,
        return_trace: bool = False,
    ) -> None:
        super().__init__(return_trace=return_trace)
        if not sql_manager.desc:
            raise ValueError("Error: sql_manager found empty description.")
        self._sql_tool = sql_manager
        self.sql_post_func = sql_post_func

        # Initialize prompts
        self._prompts = prompts or SqlCallPrompts()

        if sql_manager.db_type == "mongodb":
            self._query_prompter = ChatPrompter(instruction=self._prompts.mongodb_query_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = "mongodb json pipeline"
            self._pattern = re.compile(r"```json(.+?)```", re.DOTALL)
        else:
            self._query_prompter = ChatPrompter(instruction=self._prompts.sql_query_template).pre_hook(
                self.sql_query_promt_hook
            )
            statement_type = "sql query"
            self._pattern = re.compile(r"```sql(.+?)```", re.DOTALL)

        self._llm_query = llm.share(prompt=self._query_prompter).used_by(self._module_id)
        self._answer_prompter = ChatPrompter(
            instruction=self._prompts.db_explain_template.format(
                statement_type=statement_type, db_type=sql_manager.db_type
            )
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
        history: Optional[List[Union[List[str], Dict[str, Any]]]] = None,
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
            history or [],
            tools,
            label,
        )

    def sql_explain_prompt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],  # noqa B006
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
