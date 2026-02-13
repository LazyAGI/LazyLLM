# flake8: noqa: E501
from lazyllm.prompt_templates.prompt_library import DataPrompt


DataPrompt.add_prompt('filed_extractor', '''\
你是一个{persona}，专门从非结构化文本中提取结构化JSON。
请严格按照用户指定的字段输出，不要添加额外解释。''', '''\
文本内容：
“{text}”

请提取以下字段并以JSON格式返回：
{fields}''')

DataPrompt.add_prompt('filed_extractor', '''\
You are a {persona}, specialized in extracting structured JSON from unstructured text.
Please strictly output according to the fields specified by the user, and do not add any additional explanations.''', '''\
Text content:
“{text}”

Please extract the following fields and return them in JSON format:
{fields}''', lang='en')

DataPrompt.add_prompt('schema_extractor', '''\
你是一个高级JSON提取专家，专门从非结构化文本中提取结构化JSON。
请严格按照下面的Schema进行提取，不要添加额外解释。Schema如下：
{schema}\n''', '''\
文本内容：
“{text}”
请提取上述Schema中指定的字段并以JSON格式返回。''')

DataPrompt.add_prompt('schema_extractor', '''\
You are an advanced JSON extraction expert, specialized in extracting structured JSON from unstructured text.
Please strictly extract according to the schema below and do not add any additional explanations. Schema is as follows:
{schema}\n''', '''\
Text content:
“{text}”
Please extract the fields specified in the above schema and return them in JSON format.''', lang='en')
