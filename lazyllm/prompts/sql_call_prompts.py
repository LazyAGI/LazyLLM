# coding: utf-8

from dataclasses import dataclass
from typing import ClassVar, List


NS_TO_SQL_INSTRUCT_TEMPLATE = """
Given the following SQL tables and current date {current_date}, your job is to write sql queries in {db_type} given a user's request.

{desc}

Alert: Just reply the sql query in a code block start with triple-backticks and keyword "sql"
"""  # noqa E501


NS_TO_MONGODB_INSTRUCT_TEMPLATE = """
Current date is {current_date}.
You are a seasoned expert with 10 years of experience in crafting NoSQL queries for {db_type}. 
I will provide a collection description in a specified format. 
Your task is to analyze the user_question, which follows certain guidelines, and generate a NoSQL MongoDB aggregation pipeline accordingly.

{desc}

Note: Please return the json pipeline in a code block start with triple-backticks and keyword "json".
"""  # noqa E501


DB_RESULT_EXPLAIN_INSTRUCT_TEMPLATE = """
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


@dataclass
class SqlCallPrompts:
    """Prompt configuration class for SQL Call operations"""

    sql_query_template: str = NS_TO_SQL_INSTRUCT_TEMPLATE
    mongodb_query_template: str = NS_TO_MONGODB_INSTRUCT_TEMPLATE
    db_explain_template: str = DB_RESULT_EXPLAIN_INSTRUCT_TEMPLATE

    KEYWORDS_IN_SQL_QUERY: ClassVar[List[str]] = ["{current_date}", "{db_type}", "{desc}"]
    KEYWORDS_IN_MONGODB_QUERY: ClassVar[List[str]] = ["{current_date}", "{db_type}", "{desc}"]
    KEYWORDS_IN_DB_EXPLAIN: ClassVar[List[str]] = [
        "{{history_info}}", "{db_type}", "{{desc}}", "{statement_type}", "{{query}}", "{{result}}"]

    def __post_init__(self):
        if not isinstance(self.sql_query_template, str):
            raise ValueError("sql_query_template must be a string")

        def check_keywords(template: str, keywords: List[str]):
            for keyword in keywords:
                if keyword not in template:
                    raise ValueError(f"template must contain {keyword}")

        check_keywords(self.sql_query_template, self.KEYWORDS_IN_SQL_QUERY)
        check_keywords(self.mongodb_query_template, self.KEYWORDS_IN_MONGODB_QUERY)
        check_keywords(self.db_explain_template, self.KEYWORDS_IN_DB_EXPLAIN)
