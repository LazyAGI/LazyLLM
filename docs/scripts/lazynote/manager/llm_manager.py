from typing import Optional
from .base import BaseManager, DocstringMode
from lazynote.editor import CustomEditor
from lazyllm import OnlineChatModule, LOG
import libcst as cst
import inspect
import textwrap
import json

GENERAL_PROMPT = """请为以下代码生成注释字符串：  

目标对象：{object_name}  
代码内容：  
{node_code}  
语言：{language}  

要求：  
1. 使用简洁明了的语言描述代码功能。  
2. 方法注释需符合 Google 风格，包含功能描述、参数说明、返回值说明。  
3. 若目标对象为类，则注释需为单行简短描述，概括类的核心功能。  

注意：  
- 仅以字符串形式输出完整的注释内容，不输出原代吗！不输出原代吗！ 
- 注释格式规范，缩进与代码结构保持一致，可直接粘贴使用。  
"""


CLASS_PROMPT = """你是一个专业代码文档生成器，请为指定代码生成可直接粘贴的注释，严格遵循以下规则：

代码内容：  
{node_code}  
语言：{language}  

要求：  
1. 使用Google风格的文档字符串，语言简洁明了，准确描述代码功能。  
2. 以JSON格式输出，遵循给定格式，保持文档字符串的正确缩进和格式。  
3. 仅输出JSON文件，勿输出任何额外内容。

输出格式：
{obj_dict}

"""


class LLMDocstringManager(BaseManager):
    """
    使用大语言模型生成代码注释的管理器。
    支持模块、类和函数级别的注释生成。
    """
    llm: OnlineChatModule = None
    language: str = 'zh'
    
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
                gen_docstring=self.gen_docstring, gen_class_docstring=self.gen_class_docstring, pattern=self.pattern, module=module)
            modified_tree = tree.visit(transformer)
            self._write_code_to_file(module, modified_tree.code)
            return modified_tree.code
        except Exception as e:
            self._handle_error(f"Skipping module {module.__name__} due to error", e)
            return None
    
    def gen_class_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        """
        为类及其方法生成文档字符串。

        Args:
            old_docstring (Optional[str]): 原有的文档字符串
            node_code (str): 节点的代码内容

        Returns:
            str: 生成的文档字符串字典的JSON字符串
        """
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
                            
            language = '中文' if self.language == 'zh' else '英文'
            prompt = CLASS_PROMPT.format(
                node_code=node_code,
                language=language,
                obj_dict=json.dumps(obj_dict, indent=4, ensure_ascii=False)
            )
            

            res = self.llm(prompt)
            LOG.info(prompt)
            LOG.info(f"ORIGENIAL RES\n{res}\n")
            
            try:
                doc_dict = json.loads(res)
                if old_docstring:
                    obj_dict[class_name] = old_docstring
                return doc_dict
            except json.JSONDecodeError as e:
                LOG.info(f"Error parsing LLM response as JSON: {str(e)}")
                return {}
                
        except Exception as e:
            LOG.info(f"Error generating class docstring: {str(e)}")
            return {}
        
    def gen_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        if old_docstring:
            return old_docstring
 
        object_name = ''
        lines = node_code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('@'):
                continue
            if line.startswith('def '):
                object_name = line.split('def ')[1].split('(')[0].strip()
                break
            elif line.startswith('class '):
                object_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                break
            
        language = '中文' if self.language == 'zh' else '英文'
        prompt = GENERAL_PROMPT.format(object_name, node_code, language)
        res = self.llm(prompt)
        LOG.info(f"ORIGENIAL RES\n{res}\n")

        if '"""' in res:
            start = res.find('"""') + 3
            end = res.rfind('"""')
            if start < end:
                content = res[start:end].strip()
                lines = content.split('\n')
                formatted_lines = [lines[0]]
                formatted_lines += ['\t' + line for line in lines[1:]]
                res = '\n'.join(formatted_lines)
        return f"{res}\n\t"