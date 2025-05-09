from typing import Optional
from .base import BaseManager, DocstringMode
from lazyllm import OnlineChatModule

class LLMDocstringManager(BaseManager):
    """
    使用大语言模型生成代码注释的管理器。
    支持模块、类和函数级别的注释生成。
    """
    llm: OnlineChatModule = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern = DocstringMode.FILL
        self.llm = OnlineChatModule(source='deepseek', stream=False)
        
    def gen_docstring(self, old_docstring: Optional[str], node_code: str) -> str:
        """
        使用大语言模型生成新的文档字符串。

        Args:
            old_docstring (Optional[str]): 原有的文档字符串
            node_code (str): 节点的代码内容

        Returns:
            str: 生成的新文档字符串
        """
        # LOG.info("old_docstring: %s", old_docstring)
        prompt = self._generate_prompt(old_docstring, node_code)
        # LOG.info(f"LLM prompt: {prompt}")
        res = self.llm(prompt)

        # LOG.info(f"LLM res: {res}")
        # 提取两个三引号之间的内容
        if '"""' in res:
            start = res.find('"""') + 3
            end = res.rfind('"""')
            if start < end:
                res = res[start:end].strip()
        return res
     
    def _generate_prompt(self, old_docstring: Optional[str], node_code: str, language='中文') -> str:
        """
        根据代码内容生成提示词。

        Args:
            old_docstring (Optional[str]): 原有的文档字符串
            node_code (str): 节点的代码内容

        Returns:
            str: 生成的提示词
        """
        # 提取顶层对象名
        object_name = ''
        first_line = node_code.strip().split('\n')[0]
        if first_line.startswith('def '):
            object_name = first_line.split('def ')[1].split('(')[0].strip()
        elif first_line.startswith('class '):
            object_name = first_line.split('class ')[1].split('(')[0].split(':')[0].strip()

        return f"""请根据以下代码为目标对象生成注释字符串：

目标对象：{object_name}

代码内容：
{node_code}

原有文档字符串：
{old_docstring if old_docstring else '无'}

要求：
1. 使用清晰的语言描述代码的功能
2. 若对象为方法，生成 Google 风格的注释
3. 若对象为类，仅生成一行注释描述类的主要功能

注意：
- 仅输出生成的注释内容！也就是以三个双引号或三个单引号开头结尾的内容!
- 不输出原代码、额外解释、说明!
- 使用{language}生成注释!
"""
