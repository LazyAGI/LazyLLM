from typing import TypedDict
from typing import Union, Tuple, List, get_type_hints
import lazyllm
from lazyllm import OnlineChatModule, TrainableModule
from ..data_loaders import DirectoryReader
from .prompts import PROMPTS as DOC_KWS_PROMPTS
import random
import re
import json
import tiktoken


class DocInfoSchemaItem(TypedDict):
    key: str
    desc: str
    type: str


DocInfoSchema = List[DocInfoSchemaItem]


def validate_schema_item(given_dict: dict, typed_dict_schema: type[dict]) -> Tuple[bool, str]:
    type_hints = get_type_hints(typed_dict_schema)

    for key, value_type in type_hints.items():
        if key not in given_dict:
            return False, f"key {key} is missing"
        elif not isinstance(given_dict[key], value_type):
            return False, f"key {key} should be of type {value_type.__name__}"
    return True, "Success"


def trim_content_by_token_num(tokenizer, doc_content: str, token_limit: int):
    current_doc_token_num = len(tokenizer.encode(doc_content))
    if current_doc_token_num > token_limit:
        ratio = token_limit / current_doc_token_num
        doc_content = doc_content[: int(len(doc_content) * ratio)]
    return doc_content


class DocGenreAnalyser:
    ONE_DOC_TOKEN_LIMIT = 10000

    def __init__(self, maximum_doc_num=3):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r"```json(.+?)```", re.DOTALL)
        self._maximum_doc_num = maximum_doc_num
        self._tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        assert self._maximum_doc_num > 0

    def gen_detection_query(self, doc_path: str):
        root_nodes = self._reader.load_data([doc_path], None)
        doc_content = ""
        for root_node in root_nodes:
            doc_content += root_node.text + "\n"
        doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
        query = DOC_KWS_PROMPTS["doc_type_detection"].format(doc_content=doc_content)
        query += "\nBelow is the content of each document sample.\n\n"
        return query

    def _extract_doc_type_from_response(self, str_response: str) -> str:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                res_dict = json.loads(extracted_content)
                if not isinstance(res_dict, dict) or "doc_type" not in res_dict:
                    return ""
                return res_dict["doc_type"]
            except Exception as e:
                lazyllm.LOG.warning(f"Exception: {str(e)}, response_str: {str_response}")
                return ""
        else:
            return ""

    def analyse_doc_genre(self, llm: Union[OnlineChatModule, TrainableModule], doc_path: str) -> str:
        query = self.gen_detection_query(doc_path)
        response = llm(query)
        doc_genre = self._extract_doc_type_from_response(response)
        return doc_genre


class DocInfoSchemaAnalyser:
    ONE_DOC_TOKEN_LIMIT = 30000

    def __init__(self, maximum_doc_num=3):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r"```json(.+?)```", re.DOTALL)
        self._maximum_doc_num = maximum_doc_num
        self._tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        assert self._maximum_doc_num > 0

    def _gen_first_round_query(self, doc_type: str, doc_paths: list[str]):
        doc_contents = []
        for doc_path in doc_paths:
            root_nodes = self._reader.load_data([doc_path], None)
            doc_content = ""
            for root_node in root_nodes:
                doc_content += root_node.text + "\n"
            doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
            doc_contents.append(doc_content)
        query = DOC_KWS_PROMPTS["kws_generation"].format(number=len(doc_contents), doc_type=doc_type)
        query += "\nBelow is the content of each document sample.\n\n"
        for i, doc_content in enumerate(doc_contents):
            query += f"Document {i+1}:\n```\n{doc_content}\n```\n\n"
        return query

    def _extract_schema_from_response(self, str_response: str) -> List[dict]:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        empty_list = []
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                kws_list = json.loads(extracted_content)
                # in case of the list is in a dict, unpack it
                if isinstance(kws_list, dict):
                    values = list(kws_list.values())
                    if len(values) == 1 and isinstance(values[0], list):
                        return values[0]
                if not isinstance(kws_list, list):
                    lazyllm.LOG.warning(f"Excepted original type list but got {type(kws_list)} value: {kws_list}")
                    return empty_list
                return kws_list
            except Exception as e:
                lazyllm.LOG.warning(f"Exception: {str(e)}, response_str: {str_response}")
                return empty_list
        else:
            return empty_list

    def analyse_info_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], doc_type: str, doc_paths: list[str]
    ) -> DocInfoSchema:
        RANDOM_SEED = 1331
        if len(doc_paths) > self._maximum_doc_num:
            doc_paths.sort()
            random.seed(RANDOM_SEED)
            doc_paths = random.sample(doc_paths, self._maximum_doc_num)
        first_round_query = self._gen_first_round_query(doc_type, doc_paths)
        first_response = llm(first_round_query)
        info_schema = self._extract_schema_from_response(first_response)
        for info_schema_item in info_schema:
            is_success, msg = validate_schema_item(info_schema_item, DocInfoSchemaItem)
            if not is_success:
                lazyllm.LOG.warning(f"Please Try Again! Invalid kws dict: {info_schema_item}, error_msg: {msg}")
                return []
        return info_schema


class DocInfoExtractor:
    ONE_DOC_TOKEN_LIMIT = 50000

    def __init__(self):
        self._reader = DirectoryReader(None, {}, {})
        self._pattern = re.compile(r"```json(.+?)```", re.DOTALL)
        self._tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _gen_extraction_query(self, doc_path: str, info_schema: DocInfoSchema, extra_desc: str) -> str:
        root_nodes = self._reader.load_data([doc_path], None)
        doc_content = ""
        for root_node in root_nodes:
            doc_content += root_node.text + "\n"
        doc_content = trim_content_by_token_num(self._tiktoken_tokenizer, doc_content, self.ONE_DOC_TOKEN_LIMIT)
        if not extra_desc:
            extra_desc = f"Extra description: \n{extra_desc}"
        query = DOC_KWS_PROMPTS["kws_extraction"].format(
            kws_desc=json.dumps(info_schema), extra_desc=extra_desc, doc_content=doc_content
        )
        return query

    def _extract_kws_value_from_response(self, str_response: str) -> dict:
        # Remove the triple backticks if present
        matches = self._pattern.findall(str_response)
        empty_dict = {}
        if matches:
            # Return the first match
            extracted_content = matches[0].strip()
            try:
                kws_value = json.loads(extracted_content)
                if not isinstance(kws_value, dict):
                    lazyllm.LOG.warning(f"Excepted original type list but got {type(kws_value)}")
                    return empty_dict
                new_dict = {k: v for k, v in kws_value.items() if (isinstance(v, str) and v and v != "None")}
                return new_dict
            except Exception as e:
                lazyllm.LOG.warning(f"Exception: {str(e)}, response_str: {str_response}")
                return empty_dict
        else:
            return empty_dict

    def _format_info_by_schema(self, info: dict, info_schema: DocInfoSchema):
        valid_keys = set([info_schema_item["key"] for info_schema_item in info_schema])
        return {k: v for k, v in info.items() if k in valid_keys}

    def extract_doc_info(
        self,
        llm: Union[OnlineChatModule, TrainableModule],
        doc_path: str,
        info_schema: DocInfoSchema,
        extra_desc: str = "",
    ) -> dict:
        extraction_query = self._gen_extraction_query(doc_path, info_schema, extra_desc)
        response = llm(extraction_query)
        info: dict = self._extract_kws_value_from_response(response)
        info: dict = self._format_info_by_schema(info, info_schema)
        return info
