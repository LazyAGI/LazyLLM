import libcst as cst
import inspect
import textwrap
import json
from typing import Optional
from lazyllm import LOG, OnlineChatModule
from .base import BaseManager, DocstringMode
from lazynote.editor import CustomEditor


GENERAL_PROMPT = """你是一个专业代码文档生成器，请根据给定代码为指定目标对象生成文档字符串，严格遵循以下规则：

要求：
1. 使用{language}生成文档字符串。
2. 只为目标对象生成文档字符串，且应符合 Google 风格。
3. 若目标对象为类，仅生成单行注释描述类的核心能力。
4. 仅以字符串形式输出完整的注释内容，不包含原代吗，不包含三引号。

示例输出（函数）：
Calculates the sum of two numbers.

Args:
    a (int): First number to add
    b (int): Second number to add

Returns:
    int: Sum of the two numbers

示例输出（类）：
Calculates the sum of two numbers.

现在请为以下代码生成注释：
目标对象：{object_name}
代码内容:
{node_code}
"""


CLASS_PROMPT = """你是一个专业代码文档生成器，请为给定代码生成文档字符串，严格遵循以下规则：

要求：
1. 使用{language}生成文档字符串。
2. 使用Google风格的文档字符串，语言简洁明了，准确描述代码功能。
3. 将生成的文档字符串组织为JSON字典输出，遵循给定输出格式，保持文档字符串的正确缩进和格式, 不包含三引号。
3. 仅输出JSON字符串，勿输出任何额外内容。

代码内容：
{node_code}

输出格式：
{obj_dict}

"""


class CustomManager(BaseManager):
    """
    使用大语言模型生成代码注释的管理器。
    """

    llm: OnlineChatModule = None
    language: str = "zh"

    def __init__(self, llm, language, **kwargs):
        super().__init__(**kwargs)
        self.pattern = DocstringMode.FILL
        self.llm = llm
        self.language = language

    def modify_docstring(self, module: object) -> Optional[str]:

        try:
            source_code = inspect.getsource(module)
            source_code = textwrap.dedent(source_code)
            tree = cst.parse_module(source_code)
            transformer = CustomEditor(
                gen_docstring=self.gen_docstring,
                gen_class_docstring=self.gen_class_docstring,
                pattern=self.pattern,
                module=module,
            )
            modified_tree = tree.visit(transformer)
            self._write_code_to_file(module, modified_tree.code)
            return modified_tree.code
        except Exception as e:
            self._handle_error(f"Skipping module {module.__name__} due to error", e)
            return None

    def gen_class_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        try:
            module = cst.parse_module(node_code)
            obj_dict = {}
            for node in module.body:
                if isinstance(node, cst.ClassDef):
                    class_name = node.name.value
                    obj_dict[class_name] = ""

                    def process_class_body(class_node, parent_name):
                        for sub_node in class_node.body.body:
                            if isinstance(sub_node, cst.ClassDef):
                                sub_class_name = f"{parent_name}.{sub_node.name.value}"
                                obj_dict[sub_class_name] = ""
                                process_class_body(sub_node, sub_class_name)
                            elif isinstance(sub_node, cst.FunctionDef):
                                method_name = f"{parent_name}.{sub_node.name.value}"
                                obj_dict[method_name] = ""

                    process_class_body(node, class_name)

            language = "中文" if self.language == "zh" else "英文"
            prompt = CLASS_PROMPT.format(
                node_code=node_code,
                language=language,
                obj_dict=json.dumps(obj_dict, indent=4, ensure_ascii=False),
            )

            res = self.llm(prompt)
            def extract_json_from_response(response: str) -> dict:
                start = response.find('{')
                end = response.rfind('}')
                if start == -1 or end == -1:
                    return response
                json_str = response[start:end + 1]
                return json.loads(json_str)

            try:
                doc_dict = extract_json_from_response(res)
                doc_dict = {
                    name: self._fix_docstring_indent(
                        docstring, indent=4 * (name.count(".") + 1)
                    )
                    for name, docstring in doc_dict.items()
                }
                if old_docstring:
                    obj_dict[class_name] = old_docstring
                return doc_dict
            except json.JSONDecodeError as e:
                LOG.info(f"Error in parsing class docstring: {e}")
                return {}

        except Exception as e:
            LOG.info(f"Error in generate class docstring: {e}")
            return {}

    def gen_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        if old_docstring:
            return old_docstring

        object_name = ""
        lines = node_code.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("@"):
                continue
            if line.startswith("def "):
                object_name = line.split("def ")[1].split("(")[0].strip()
                break
            elif line.startswith("class "):
                object_name = (
                    line.split("class ")[1].split("(")[0].split(":")[0].strip()
                )
                break
        language = "中文" if self.language == "zh" else "英文"
        prompt = GENERAL_PROMPT.format(
            object_name=object_name, node_code=node_code, language=language
        )
        res = self.llm(prompt)
        return self._fix_docstring_indent(res)

    def _fix_docstring_indent(self, docstring, indent: int = 4):
        if not docstring or not indent:
            return ""

        lines = docstring.strip().split("\n")
        if len(lines) <= 1:
            return docstring.strip()

        def get_indent(line):
            return len(line) - len(line.lstrip())

        non_empty_lines = [line for line in lines[1:] if line.strip()]
        if not non_empty_lines:
            return lines[0]
        min_indent = min(get_indent(line) for line in non_empty_lines)

        result = [lines[0].strip()]
        for line in lines[1:]:
            if line.strip():
                result.append(" " * indent + line[min_indent:])
            else:
                result.append(" " * indent)
        return "\n".join(result)
